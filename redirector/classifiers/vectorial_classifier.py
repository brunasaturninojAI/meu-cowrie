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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

logger = logging.getLogger(__name__)

class VectorialClassifier:
    """
    Cria classificadores binários usando vetores TF-IDF de descrições de comandos.
    Segue a abordagem do SentiWordNet de usar dois classificadores binários por traço.
    """
    
    def __init__(self, gloss_extractor):
        self.gloss_extractor = gloss_extractor
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Aumentado de 1000 para 2000
            stop_words='english',
            ngram_range=(1, 3),  # Aumentado de (1,2) para (1,3)
            min_df=1,  # Reduzido de 2 para 1 para capturar mais variações
            max_df=0.95,  # Adicionado para filtrar termos muito comuns
            sublinear_tf=True,  # Adicionado para melhor normalização
            use_idf=True,
            smooth_idf=True
        )
        self.classifiers = {}
        
    def prepare_global_vectorizer(self, all_known_commands: List[str]):
        """
        Treina o vectorizer TF-IDF com TODOS os comandos conhecidos.
        Deve ser chamado antes de train_trait_classifiers.
        """
        logger.info(f"Preparando vectorizer global com {len(all_known_commands)} comandos...")
        
        # Obtém glossas para todos os comandos conhecidos
        all_glosses = []
        for cmd in all_known_commands:
            try:
                gloss = self.gloss_extractor.get_command_gloss(cmd)
                all_glosses.append(gloss)
            except Exception as e:
                logger.warning(f"Erro ao obter gloss para '{cmd}': {e}")
                all_glosses.append(f"system command: {cmd}")
        
        # Treina o vectorizer com todo o vocabulário
        self.vectorizer.fit(all_glosses)
        logger.info(f"Vectorizer treinado com vocabulário de {len(self.vectorizer.vocabulary_)} features")

    def train_trait_classifiers(self, expanded_sets: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        Treina classificadores binários para cada traço.
        ATENÇÃO: prepare_global_vectorizer() deve ser chamado primeiro!
        Retorna dicionário de pares (classificador_positivo, classificador_negativo).
        """
        if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
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
            pos_glosses = [self.gloss_extractor.get_command_gloss(cmd) for cmd in positive_commands]
            neg_glosses = [self.gloss_extractor.get_command_gloss(cmd) for cmd in negative_commands]
            obj_glosses = [self.gloss_extractor.get_command_gloss(cmd) for cmd in objective_commands]
            
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
            clf_positive = SVC(
                probability=True, 
                random_state=42,
                kernel='rbf',
                C=1.0,  # Parâmetro de regularização
                gamma='scale',  # Kernel coefficient
                class_weight='balanced'  # Para lidar com classes desbalanceadas
            )
            clf_negative = SVC(
                probability=True, 
                random_state=42,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced'
            )
            
            clf_positive.fit(X, y_positive)
            clf_negative.fit(X, y_negative)
            
            # Validação cruzada mais robusta com StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            pos_score = cross_val_score(clf_positive, X, y_positive, cv=cv, scoring='f1_weighted').mean()
            neg_score = cross_val_score(clf_negative, X, y_negative, cv=cv, scoring='f1_weighted').mean()
            
            logger.info(f"  {trait} positive classifier CV score (F1-weighted): {pos_score:.3f}")
            logger.info(f"  {trait} negative classifier CV score (F1-weighted): {neg_score:.3f}")
            
            trait_classifiers[trait] = (clf_positive, clf_negative)
        
        return trait_classifiers
    
    def _get_objective_commands(self, labeled_commands: Set[str], max_objective: int = 50) -> List[str]:
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
            
            # Comandos modernos neutros - Docker
            "docker --version", "docker info", "docker version",
            "docker images", "docker ps", "docker logs",
            "docker stats", "docker system df", "docker network ls",
            
            # Comandos modernos neutros - Kubernetes  
            "kubectl version", "kubectl config view", "kubectl cluster-info",
            "kubectl get namespaces", "kubectl get nodes", "kubectl api-versions",
            
            # Comandos modernos neutros - Sistema
            "systemctl --version", "systemctl list-unit-files",
            "systemctl get-default", "systemctl status",
            "journalctl --version", "loginctl list-sessions",
            
            # Comandos modernos neutros - Git
            "git --version", "git config --list", "git remote -v",
            "git branch", "git tag", "git log --oneline",
            
            # Comandos modernos neutros - Package managers
            "npm --version", "npm list", "npm config list",
            "pip --version", "pip list", "pip show",
            "conda --version", "conda list", "conda info",
            
            # Comandos modernos neutros - Cloud
            "aws --version", "gcloud version", "az version",
            "terraform --version", "ansible --version",
            
            # Comandos de texto neutros
            "wc", "sort", "uniq", "cut", "paste", "join",
            "tr", "expand", "unexpand", "fmt", "fold",
            
            # Comandos de arquivo neutros
            "find . -name", "locate", "stat", "file", "basename",
            "dirname", "realpath", "readlink", "ln -s",
            
            # Comandos de processo neutros
            "ps aux", "pgrep", "pidof", "jobs", "nohup",
            "screen -list", "tmux list-sessions",
            
            # Comandos de compressão neutros
            "tar -tf", "zip -l", "unzip -l", "gzip -l",
            "file", "xxd", "hexdump", "strings",
        ]
        
        objective = [cmd for cmd in common_commands if cmd not in labeled_commands]
        return objective[:max_objective]
    
    def classify_command(self, command: str, trait_classifiers: Dict[str, Tuple]) -> Dict[str, Dict[str, float]]:
        """
        Classifica um comando usando classificadores treinados.
        Retorna pontuações para cada traço: {trait: {"positive": p, "negative": n, "objective": o}}
        """
        gloss = self.gloss_extractor.get_command_gloss(command)
        X = self.vectorizer.transform([gloss])
        
        results = {}
        
        for trait, (clf_pos, clf_neg) in trait_classifiers.items():
            # Obtém probabilidades de ambos os classificadores
            pos_proba = clf_pos.predict_proba(X)[0][1]  # probabilidade de ser positivo
            neg_proba = clf_neg.predict_proba(X)[0][1]  # probabilidade de ser negativo
            
            # Lógica de decisão melhorada e menos polarizada
            # Threshold mais baixo para reduzir polarização
            threshold = 0.4  # Reduzido de 0.5 para 0.4
            
            if pos_proba > threshold and neg_proba <= threshold:
                # Positivo - mas com score mais suave
                positive_score = min(0.8, pos_proba)  # Limita score máximo
                negative_score = max(0.1, neg_proba * 0.5)  # Score negativo mínimo
                objective_score = 1.0 - positive_score - negative_score
            elif pos_proba <= threshold and neg_proba > threshold:
                # Negativo - mas com score mais suave
                positive_score = max(0.1, pos_proba * 0.5)  # Score positivo mínimo
                negative_score = min(0.8, neg_proba)  # Limita score máximo
                objective_score = 1.0 - positive_score - negative_score
            else:
                # Objetivo ou ambíguo - scores mais balanceados
                positive_score = max(0.1, pos_proba * 0.6)
                negative_score = max(0.1, neg_proba * 0.6)
                objective_score = 1.0 - positive_score - negative_score
            
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
            "vectorizer_features": len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0,
            "trained_classifiers": len(self.classifiers),
            "vectorizer_params": {
                "max_features": self.vectorizer.max_features,
                "ngram_range": self.vectorizer.ngram_range,
                "min_df": self.vectorizer.min_df,
                "max_df": self.vectorizer.max_df
            }
        } 