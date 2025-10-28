#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vectorial_classifier.py
Cria classificadores binários usando vetores TF-IDF de descrições de comandos.
Segue a abordagem do SentiWordNet de usar dois classificadores binários por traço.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from scipy import sparse
import re
import logging

logger = logging.getLogger(__name__)

class VectorialClassifier:
    """
    Cria classificadores binários usando vetores TF-IDF de descrições de comandos.
    Segue a abordagem do SentiWordNet de usar dois classificadores binários por traço.
    """
    
    def __init__(self, gloss_extractor):
        self.gloss_extractor = gloss_extractor
        # Vetorização enriquecida: união de n-gramas de palavras e caracteres
        self.word_vectorizer = TfidfVectorizer(
            max_features=1000,  # Reduzido de 2000 para reduzir overfitting
            stop_words='english',
            ngram_range=(1, 2),  # Reduzido de (1,3) para (1,2)
            min_df=2,  # Aumentado de 1 para 2
            max_df=0.85,  # Reduzido de 0.95 para 0.85
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 4),  # Reduzido de (3,5) para (3,4)
            min_df=2,  # Aumentado de 1 para 2
            max_df=0.90,  # Reduzido de 1.0 para 0.90
            sublinear_tf=True
        )
        # Extrator de features estruturais
        self.structural_transformer = FunctionTransformer(
            func=self._extract_structural_features_batch,
            validate=False
        )
        # Union combina TF-IDF textual (features estruturais desabilitadas para reduzir overfitting)
        self.vectorizer = FeatureUnion([
            ("word", self.word_vectorizer),
            ("char", self.char_vectorizer),
            # ("structural", self.structural_transformer),  # Desabilitado temporariamente
        ])
        self.classifiers = {}
        self.relations: Dict = None
        self._is_vectorizer_fitted: bool = False
        self._feature_count: int = 0
        self._command_to_structural_cache: Dict[str, np.ndarray] = {}

    def set_relations(self, relations: Dict):
        """Configura o dicionário de relações para enriquecer as glossas."""
        self.relations = relations
        
    def prepare_global_vectorizer(self, all_known_commands: List[str]):
        """
        Treina o vectorizer TF-IDF com TODOS os comandos conhecidos.
        Deve ser chamado antes de train_trait_classifiers.
        """
        logger.info(f"Preparando vectorizer global com {len(all_known_commands)} comandos...")
        
        # Obtém glossas enriquecidas para todos os comandos conhecidos
        all_glosses = []
        for cmd in all_known_commands:
            try:
                gloss = self._get_enriched_gloss(cmd)
                all_glosses.append(gloss)
            except Exception as e:
                logger.warning(f"Erro ao obter gloss para '{cmd}': {e}")
                all_glosses.append(f"system command: {cmd}")

        # Treina o vectorizer com todo o vocabulário
        self.vectorizer.fit(all_glosses)
        # Estima a contagem total de features somando vocabulários dos componentes
        try:
            self._feature_count = (
                (len(self.word_vectorizer.vocabulary_) if hasattr(self.word_vectorizer, 'vocabulary_') and self.word_vectorizer.vocabulary_ is not None else 0)
                + (len(self.char_vectorizer.vocabulary_) if hasattr(self.char_vectorizer, 'vocabulary_') and self.char_vectorizer.vocabulary_ is not None else 0)
            )
        except Exception:
            self._feature_count = 0
        self._is_vectorizer_fitted = True
        logger.info(f"Vectorizer treinado com vocabulário combinado de {self._feature_count} features")

    def train_trait_classifiers(self, expanded_sets: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        Treina classificadores binários para cada traço.
        ATENÇÃO: prepare_global_vectorizer() deve ser chamado primeiro!
        Retorna dicionário de pares (classificador_positivo, classificador_negativo).
        """
        if not self._is_vectorizer_fitted:
            raise ValueError("Vectorizer não foi treinado! Chame prepare_global_vectorizer() primeiro.")
        
        trait_classifiers = {}
        
        # Agrupa por traço (ignorando polaridade)
        traits = set()
        for trait_polarity in expanded_sets.keys():
            trait = trait_polarity.split('_')[0]  # "Perfectionism_Positive" -> "Perfectionism"
            traits.add(trait)
        
        for trait in traits:
            logger.info(f"Training classifiers for {trait}")
            
            pos_key = f"{trait}_Positive"
            neg_key = f"{trait}_Negative"
            
            if pos_key not in expanded_sets or neg_key not in expanded_sets:
                logger.warning(f"Missing data for {trait}, skipping")
                continue
            
            positive_commands = list(expanded_sets[pos_key])
            negative_commands = list(expanded_sets[neg_key])
            
            # Cria conjunto objetivo (comandos que não estão em positivo ou negativo)
            all_commands = set(positive_commands + negative_commands)
            objective_commands = self._get_objective_commands(all_commands)
            
            # Obtém glossas (descrições) para vetorização
            pos_glosses = [self._get_enriched_gloss(cmd) for cmd in positive_commands]
            neg_glosses = [self._get_enriched_gloss(cmd) for cmd in negative_commands]
            obj_glosses = [self._get_enriched_gloss(cmd) for cmd in objective_commands]
            
            # Prepara dados de treinamento
            all_glosses = pos_glosses + neg_glosses + obj_glosses
            
            # USA o vectorizer já treinado (não chama fit novamente!)
            X = self.vectorizer.transform(all_glosses)
            
            # Prepara rótulos para classificadores binários
            # Classificador 1: Positivo vs (Negativo + Objetivo)
            y_positive = ([1] * len(pos_glosses) + 
                         [0] * len(neg_glosses) + 
                         [0] * len(obj_glosses))
            
            # Classificador 2: Negativo vs (Positivo + Objetivo)  
            y_negative = ([0] * len(pos_glosses) + 
                         [1] * len(neg_glosses) + 
                         [0] * len(obj_glosses))
            
            # Treina classificadores binários com parâmetros otimizados
            base_pos = LogisticRegression(
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=2000,  # Aumentado para melhor convergência
                C=0.5,  # Regularização aumentada para reduzir overfitting
            )
            base_neg = LogisticRegression(
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=2000,  # Aumentado para melhor convergência
                C=0.5,  # Regularização aumentada para reduzir overfitting
            )
            # Calibração com sigmoid (Platt) para preservar ordenação (melhor AUC)
            # Define n_splits dinamicamente garantindo pelo menos 2 e no máximo 5 dobras
            def _n_splits_for(y: List[int]) -> int:
                try:
                    counts = np.bincount(np.array(y, dtype=int))
                    min_class = int(counts[counts > 0].min()) if counts.size > 0 else 2
                    return max(2, min(5, min_class))
                except Exception:
                    return 3
            cv_pos_splits = _n_splits_for(y_positive)
            cv_neg_splits = _n_splits_for(y_negative)
            cv_pos = StratifiedKFold(n_splits=cv_pos_splits, shuffle=True, random_state=42)
            cv_neg = StratifiedKFold(n_splits=cv_neg_splits, shuffle=True, random_state=42)
            clf_positive = CalibratedClassifierCV(estimator=base_pos, method='sigmoid', cv=cv_pos)
            clf_negative = CalibratedClassifierCV(estimator=base_neg, method='sigmoid', cv=cv_neg)
            
            clf_positive.fit(X, y_positive)
            clf_negative.fit(X, y_negative)
            
            # Validação cruzada mais robusta com StratifiedKFold usando mesmo n_splits
            cv_eval_pos = StratifiedKFold(n_splits=cv_pos_splits, shuffle=True, random_state=42)
            cv_eval_neg = StratifiedKFold(n_splits=cv_neg_splits, shuffle=True, random_state=42)
            
            pos_score = cross_val_score(clf_positive, X, y_positive, cv=cv_eval_pos, scoring='f1_weighted').mean()
            neg_score = cross_val_score(clf_negative, X, y_negative, cv=cv_eval_neg, scoring='f1_weighted').mean()
            
            logger.info(f"  {trait} positive classifier CV score (F1-weighted): {pos_score:.3f}")
            logger.info(f"  {trait} negative classifier CV score (F1-weighted): {neg_score:.3f}")
            
            trait_classifiers[trait] = (clf_positive, clf_negative)
        
        return trait_classifiers
    
    def _get_objective_commands(self, labeled_commands: Set[str], max_objective: int = 150) -> List[str]:
        """Obtém comandos objetivos (neutros) que não estão nos conjuntos positivo/negativo."""
        common_commands = [
            # Comandos básicos tradicionais
            "ls", "pwd", "cd", "echo", "cat", "less", "more", "head", "tail",
            "cp", "mv", "mkdir", "touch", "which", "whereis", "file",
            "date", "uptime", "who", "w", "id", "groups", "env",
            
            # Comandos de informação (neutros)
            "uname", "hostname", "whoami", "history", "type",
            "man", "info", "whatis", "apropos", "locale", "printenv",
            "df", "du", "free", "lscpu", "lsblk", "lsusb", "lspci",
            
            # Comandos de rede neutros
            "ping", "host", "nslookup", "dig", "ifconfig", "ip addr",
            "route", "arp", "netstat -r", "ss -l", "hostname -I",
            "curl --help", "wget --help", "ssh -V",
            
            # Comandos modernos neutros - Docker
            "docker --version", "docker info", "docker version",
            "docker images", "docker ps", "docker logs",
            "docker stats", "docker system df", "docker network ls",
            
            # Comandos modernos neutros - Kubernetes  
            "kubectl version", "kubectl config view", "kubectl cluster-info",
            "kubectl get namespaces", "kubectl get nodes", "kubectl api-versions",
            "kubectl get pods -A", "kubectl get svc -A",
            
            # Comandos modernos neutros - Sistema
            "systemctl --version", "systemctl list-unit-files",
            "systemctl get-default", "systemctl status",
            "journalctl --version", "loginctl list-sessions",
            "timedatectl", "hostnamectl",
            
            # Comandos modernos neutros - Git
            "git --version", "git config --list", "git remote -v",
            "git branch", "git tag", "git log --oneline",
            "git rev-parse --short HEAD",
            
            # Comandos modernos neutros - Package managers
            "npm --version", "npm list", "npm config list",
            "pip --version", "pip list", "pip show",
            "conda --version", "conda list", "conda info",
            "pip freeze", "yarn --version",
            
            # Comandos modernos neutros - Cloud
            "aws --version", "gcloud version", "az version",
            "terraform --version", "ansible --version",
            "aws sts get-caller-identity",
            
            # Comandos de texto neutros
            "wc", "sort", "uniq", "cut", "paste", "join",
            "tr", "expand", "unexpand", "fmt", "fold",
            
            # Comandos de arquivo neutros
            "find . -name", "locate", "stat", "file", "basename",
            "dirname", "realpath", "readlink", "ln -s",
            "tree -L 1",
            
            # Comandos de processo neutros
            "ps aux", "pgrep", "pidof", "jobs", "nohup",
            "screen -list", "tmux list-sessions",
            "uptime", "who",
            
            # Comandos de compressão neutros
            "tar -tf", "zip -l", "unzip -l", "gzip -l",
            "file", "xxd", "hexdump", "strings",
            "bzip2 -t", "xz -t",
        ]
        
        objective = [cmd for cmd in common_commands if cmd not in labeled_commands]
        return objective[:max_objective]
    
    def classify_command(self, command: str, trait_classifiers: Dict[str, Tuple]) -> Dict[str, Dict[str, float]]:
        """
        Classifica um comando usando classificadores treinados.
        Retorna pontuações para cada traço: {trait: {"positive": p, "negative": n, "objective": o}}
        """
        gloss = self._get_enriched_gloss(command)
        X = self.vectorizer.transform([gloss])
        
        results = {}
        
        for trait, (clf_pos, clf_neg) in trait_classifiers.items():
            # Probabilidades calibradas dos classificadores binários
            pos_proba = float(clf_pos.predict_proba(X)[0][1])
            neg_proba = float(clf_neg.predict_proba(X)[0][1])

            # Combinação suave com inibição mútua
            positive_score = max(0.0, pos_proba * (1.0 - neg_proba))
            negative_score = max(0.0, neg_proba * (1.0 - pos_proba))
            objective_score = max(0.0, 1.0 - positive_score - negative_score)
            
            # Garante que todos os scores sejam positivos e somem 1.0
            positive_score = max(0.0, positive_score)
            negative_score = max(0.0, negative_score)
            objective_score = max(0.0, objective_score)
            
            # Normaliza para somar 1.0
            total = positive_score + negative_score + objective_score
            if total > 0:
                results[trait] = {
                    "positive": positive_score / total,
                    "negative": negative_score / total,
                    "objective": objective_score / total
                }
            else:
                results[trait] = {"positive": 0.33, "negative": 0.33, "objective": 0.34}
        
        return results
    
    def get_classifier_info(self) -> Dict:
        """Retorna informações sobre os classificadores treinados."""
        return {
            "vectorizer_features": self._feature_count,
            "trained_classifiers": len(self.classifiers),
            "vectorizer_params": {
                "word": {
                    "max_features": getattr(self.word_vectorizer, 'max_features', None),
                    "ngram_range": getattr(self.word_vectorizer, 'ngram_range', None),
                    "min_df": getattr(self.word_vectorizer, 'min_df', None),
                    "max_df": getattr(self.word_vectorizer, 'max_df', None),
                },
                "char": {
                    "ngram_range": getattr(self.char_vectorizer, 'ngram_range', None)
                }
            }
        }

    def _extract_structural_features(self, command: str) -> np.ndarray:
        """Extrai features estruturais de um comando.

        Retorna vetor numpy com features como:
        - command_length: comprimento do comando
        - num_args: número de argumentos
        - num_flags: número de flags (-x, --flag)
        - num_pipes: número de pipes (|)
        - num_redirects: número de redirects (>, <, >>)
        - num_logical_ops: operadores lógicos (&, &&, ||, ;)
        - has_sudo: presença de sudo/su
        - has_wildcards: presença de wildcards (*)
        - has_dangerous_patterns: palavras perigosas (rm, dd, kill, etc.)
        - num_special_chars: número de caracteres especiais
        """
        features = []

        # 1. Comprimento do comando
        features.append(float(len(command)))

        # 2. Número de argumentos (split por espaço)
        tokens = command.split()
        num_args = max(0, len(tokens) - 1)  # -1 para excluir o comando base
        features.append(float(num_args))

        # 3. Número de flags (-x, --flag)
        num_flags = len(re.findall(r'-+\w+', command))
        features.append(float(num_flags))

        # 4. Número de pipes
        num_pipes = command.count('|')
        features.append(float(num_pipes))

        # 5. Número de redirects
        num_redirects = command.count('>') + command.count('<')
        features.append(float(num_redirects))

        # 6. Operadores lógicos
        num_logical = command.count('&&') + command.count('||') + command.count(';') + command.count('&')
        features.append(float(num_logical))

        # 7. Presença de sudo/su (elevação de privilégios)
        has_sudo = 1.0 if re.search(r'\b(sudo|su)\b', command) else 0.0
        features.append(has_sudo)

        # 8. Presença de wildcards
        has_wildcards = 1.0 if '*' in command or '?' in command else 0.0
        features.append(has_wildcards)

        # 9. Presença de padrões perigosos
        dangerous_patterns = [
            r'\brm\s+-rf\b', r'\bdd\b', r'\bkill\b', r'\bkillall\b',
            r'\bchmod\s+777\b', r'\bmkfs\b', r'\bformat\b', r'\bshred\b',
            r'\bwipefs\b', r'\biptables.*-F\b', r'\bsetenforce\s+0\b'
        ]
        has_dangerous = 1.0 if any(re.search(pat, command) for pat in dangerous_patterns) else 0.0
        features.append(has_dangerous)

        # 10. Número de caracteres especiais
        special_chars = r'[!@#$%^&*(){}[\]\\|;:\'"<>?/~`]'
        num_special = len(re.findall(special_chars, command))
        features.append(float(num_special))

        # 11. Presença de network commands
        network_patterns = [r'\b(curl|wget|ssh|scp|ftp|telnet|nc|netcat)\b']
        has_network = 1.0 if any(re.search(pat, command) for pat in network_patterns) else 0.0
        features.append(has_network)

        # 12. Presença de file manipulation
        file_patterns = [r'\b(cp|mv|rm|mkdir|rmdir|touch|ln)\b']
        has_file_ops = 1.0 if any(re.search(pat, command) for pat in file_patterns) else 0.0
        features.append(has_file_ops)

        return np.array(features, dtype=float)

    def _extract_structural_features_batch(self, glosses: List[str]) -> sparse.csr_matrix:
        """Extrai features estruturais para um batch de glossas.

        FeatureUnion chama isso com as glossas, mas precisamos dos comandos originais.
        Como workaround, extraímos o comando do início da glossa.
        """
        features_list = []
        for gloss in glosses:
            # Extrai comando da primeira linha da glossa (antes de "similar", "derived", etc.)
            cmd = gloss.split('\n')[0].strip()
            # Remove prefixo padrão "system command: " se existir
            if cmd.startswith("system command: "):
                cmd = cmd.replace("system command: ", "")

            # Tenta usar cache primeiro
            if cmd in self._command_to_structural_cache:
                features_list.append(self._command_to_structural_cache[cmd])
            else:
                feats = self._extract_structural_features(cmd)
                self._command_to_structural_cache[cmd] = feats
                features_list.append(feats)

        # Converte para matriz densa e depois para sparse (para compatibilidade com FeatureUnion)
        dense_matrix = np.vstack(features_list)
        return sparse.csr_matrix(dense_matrix)

    def _get_enriched_gloss(self, command: str) -> str:
        """Retorna a glossa base enriquecida com contexto das relações taxonômicas."""
        base_gloss = self.gloss_extractor.get_command_gloss(command)
        if not self.relations:
            return base_gloss

        similar_neighbors: Set[str] = set()
        derived_neighbors: Set[str] = set()
        category_neighbors: Set[str] = set()
        antonym_neighbors: Set[str] = set()

        # Similar
        for base, similars in self.relations.get("similar", {}).items():
            if command.startswith(base) or command in similars:
                similar_neighbors.update(similars)
                similar_neighbors.add(base)

        # Derived from
        for base, derivatives in self.relations.get("derived_from", {}).items():
            if command.startswith(base) or command in derivatives:
                derived_neighbors.update(derivatives)
                derived_neighbors.add(base)

        # Categories (also_see)
        for _, cmds in self.relations.get("also_see", {}).items():
            if command in cmds:
                category_neighbors.update(cmds)

        # Antonyms
        for base, opposites in self.relations.get("antonym", {}).items():
            if command.startswith(base) or command in opposites:
                antonym_neighbors.update(opposites)
                antonym_neighbors.add(base)

        # Limitar quantidade para não poluir o texto
        def take_some(values: Set[str], max_n: int = 10) -> List[str]:
            return list(values)[:max_n]

        parts = [base_gloss]
        if similar_neighbors:
            parts.append("similar " + " ".join(take_some(similar_neighbors)))
        if derived_neighbors:
            parts.append("derived " + " ".join(take_some(derived_neighbors)))
        if category_neighbors:
            # Evita incluir o próprio comando várias vezes
            cat_tokens = [c for c in take_some(category_neighbors) if c != command]
            if cat_tokens:
                parts.append("category " + " ".join(cat_tokens))
        if antonym_neighbors:
            parts.append("antonym " + " ".join(take_some(antonym_neighbors)))

        return " \n ".join(parts)