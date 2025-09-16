#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
semisupervised_expansion.py
Expansão semi-supervisionada de classificação de comandos seguindo a metodologia SentiWordNet.

Implementa:
1. Conjuntos semente (L_p, L_n, L_o) - comandos rotulados manualmente
2. Expansão iterativa usando relações taxonômicas  
3. Vetorização TF-IDF de descrições de comandos (páginas man)
4. Classificadores binários para detecção de traços
5. Rotulação automática de todo o espaço de comandos
"""

import json
import subprocess
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONJUNTOS SEMENTE EXPANDIDOS (mais comandos para melhor treinamento) ---
SEED_SETS = {
    "Perfectionism_Positive": [
        "dmesg", "fsck", "diff", "strace", "lsof", "rpm -V", 
        "badblocks", "e2fsck", "getconf", "lscpu", "ethtool",
        "chmod", "chown", "umask", "setfacl", "getfacl", "stat",
        "file", "md5sum", "sha256sum", "cksum", "sum", "wc",
        "sort", "uniq", "comm", "diff3", "patch", "rsync",
        "tar", "gzip", "bzip2", "xz", "zip", "unzip"
    ],
    "Perfectionism_Negative": [
        "rm -rf", "kill -9", "iptables -F", "rpm --nodeeps", 
        "nohup", "shutdown -h now", "pkill", "killall",
        "dd if=/dev/zero", "mkfs.ext4", "fdisk", "parted",
        "chmod 777", "chmod -R 777", "chown -R root:root",
        "setenforce 0", "systemctl disable", "service stop"
    ],
    "Patience_Positive": [
        "watch", "tcpdump", "tail -f", "man", "apropos", 
        "sleep", "wait", "ping", "traceroute", "mtr",
        "iotop", "htop", "top", "vmstat", "iostat", "sar",
        "netstat", "ss", "lsof", "fuser", "lsof -i",
        "strace -f", "ltrace", "gdb", "valgrind", "perf"
    ],
    "Patience_Negative": [
        "pkill", "kill -9", "ctrl+c", "timeout", "pkill -9",
        "killall -9", "xkill", "kill -SIGKILL", "kill -TERM",
        "systemctl kill", "service kill", "pkill -f"
    ],
    "Creativity_Positive": [
        "gcc", "python", "git", "curl", "wget", "nc", "ssh",
        "vim", "emacs", "make", "cmake", "autoconf", "automake",
        "docker", "kubectl", "ansible", "terraform", "vagrant",
        "node", "npm", "pip", "cargo", "go", "rustc", "javac",
        "gdb", "valgrind", "perf", "strace", "ltrace", "gprof"
    ],
    "Creativity_Negative": [
        "cat", "ls", "pwd", "cd", "echo", "printf", "seq",
        "yes", "true", "false", ":", "test", "[", "[[",
        "exit", "logout", "clear", "reset", "history"
    ]
}

# --- RELAÇÕES TAXONÔMICAS EXPANDIDAS (mais comandos para melhor expansão) ---
COMMAND_RELATIONS = {
    # Funcionalidade similar (como "similar-to" do WordNet)
    "similar": {
        "ps": ["top", "htop", "pstree", "pgrep", "pidof"],
        "ls": ["dir", "ll", "tree", "find", "locate"],
        "cat": ["more", "less", "head", "tail", "tac"],
        "grep": ["egrep", "fgrep", "ag", "rg", "ack"],
        "vi": ["vim", "nano", "emacs", "gedit", "kate"],
        "rm": ["rmdir", "unlink", "shred", "wipe", "secure-delete"],
        "cp": ["mv", "rsync", "dd", "tar", "cpio"],
        "chmod": ["chown", "chgrp", "setfacl", "getfacl", "umask"],
        "mount": ["umount", "mountpoint", "findmnt", "blkid", "lsblk"],
        "ping": ["traceroute", "mtr", "fping", "nping", "hping3"],
        "netstat": ["ss", "lsof", "fuser", "netstat", "ip"],
        "gcc": ["g++", "clang", "clang++", "icc", "pgcc"]
    },
    
    # Funcionalidade oposta (como "antonym" do WordNet)
    "antonym": {
        "start": ["stop", "kill", "terminate", "end", "halt"],
        "mount": ["umount", "unmount", "dismount"],
        "compress": ["decompress", "uncompress", "extract", "unzip", "untar"],
        "encrypt": ["decrypt", "decipher", "decode"],
        "connect": ["disconnect", "close", "shutdown", "exit"],
        "enable": ["disable", "block", "prevent", "deny"],
        "install": ["uninstall", "remove", "purge", "erase"],
        "create": ["delete", "destroy", "remove", "erase"]
    },
    
    # Relações de ferramentas (como "derived-from" do WordNet)
    "derived_from": {
        "gcc": ["g++", "gdb", "gprof", "gcov", "gcc-ar"],
        "git": ["gitk", "git-log", "git-gui", "git-citool", "gitk"],
        "docker": ["docker-compose", "docker-machine", "docker-swarm", "kubectl"],
        "ssh": ["scp", "sftp", "ssh-keygen", "ssh-agent", "ssh-add"],
        "python": ["pip", "conda", "poetry", "pipenv", "pyenv"],
        "node": ["npm", "yarn", "pnpm", "npx", "node-gyp"],
        "vim": ["gvim", "vimdiff", "vimtutor", "xxd", "vim"]
    },
    
    # Categorias funcionais (como "also-see" do WordNet)
    "also_see": {
        "network_analysis": ["netstat", "ss", "lsof", "tcpdump", "wireshark", "nmap", "ncat", "netcat"],
        "process_monitoring": ["ps", "top", "htop", "pstree", "pgrep", "pidof", "kill", "killall"],
        "file_operations": ["ls", "find", "locate", "which", "whereis", "file", "stat", "touch"],
        "system_info": ["uname", "lscpu", "lsblk", "df", "du", "free", "uptime", "who"],
        "text_processing": ["grep", "sed", "awk", "cut", "paste", "join", "sort", "uniq"],
        "compression": ["gzip", "bzip2", "xz", "zip", "tar", "7z", "rar", "lzma"],
        "security": ["chmod", "chown", "umask", "setfacl", "getfacl", "passwd", "su", "sudo"],
        "development": ["gcc", "make", "cmake", "git", "vim", "emacs", "gdb", "valgrind"]
    }
}

class CommandGlossExtractor:
    """Extrai 'glossas' (descrições) das páginas man, similar às glossas do WordNet."""
    
    def __init__(self):
        self.cache = {}
    
    def get_command_gloss(self, command: str) -> str:
        """Extrai descrição da página man (equivalente à glossa do WordNet)."""
        if command in self.cache:
            return self.cache[command]
        
        # Processa comandos compostos
        base_cmd = command.split()[0].replace('-', '_')
        
        try:
            # Tenta obter a página man
            result = subprocess.run(
                ["man", base_cmd], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                gloss = self._extract_description(result.stdout)
            else:
                # Fallback para descrição breve
                gloss = self._get_fallback_description(command)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            gloss = self._get_fallback_description(command)
        
        self.cache[command] = gloss
        return gloss
    
    def _extract_description(self, man_output: str) -> str:
        """Extrai a seção DESCRIPTION da página man."""
        lines = man_output.split('\n')
        description_lines = []
        in_description = False
        
        for line in lines:
            if re.match(r'^DESCRIPTION', line.strip()):
                in_description = True
                continue
            elif re.match(r'^[A-Z][A-Z\s]+$', line.strip()) and in_description:
                break  # Próxima seção
            elif in_description and line.strip():
                description_lines.append(line.strip())
        
        if description_lines:
            return ' '.join(description_lines[:10])  # Primeiras 10 linhas
        else:
            return f"system command: {man_output.split(chr(10))[0] if man_output else 'unknown'}"
    
    def _get_fallback_description(self, command: str) -> str:
        """Descrições de fallback para comandos comuns."""
        fallbacks = {
            "rm -rf": "forcefully remove files and directories recursively without confirmation",
            "kill -9": "terminate process immediately using SIGKILL signal",
            "iptables -F": "flush all firewall rules removing network security",
            "rpm --nodeeps": "install package ignoring dependency requirements",
            "watch": "execute command repeatedly displaying output updates",
            "tcpdump": "capture and analyze network packets in real time",
            "dmesg": "display kernel ring buffer messages for system diagnostics",
            "fsck": "check and repair filesystem integrity and consistency",
        }
        
        return fallbacks.get(command, f"system command: {command}")

class SemiSupervisedExpander:
    """
    Expansão semi-supervisionada seguindo a metodologia SentiWordNet.
    Expande conjuntos semente usando relações taxonômicas iterativamente.
    """
    
    def __init__(self, seed_sets: Dict[str, List[str]], relations: Dict[str, Dict], max_iterations: int = 3):
        self.seed_sets = seed_sets
        self.relations = relations
        self.max_iterations = max_iterations
        self.gloss_extractor = CommandGlossExtractor()
        
    def expand_seeds(self) -> Dict[str, Set[str]]:
        """
        Expande iterativamente conjuntos semente usando relações taxonômicas.
        Retorna conjuntos de treinamento expandidos finais.
        """
        expanded_sets = {}
        
        for trait_polarity, initial_seeds in self.seed_sets.items():
            logger.info(f"Expanding {trait_polarity}: {len(initial_seeds)} initial seeds")
            
            current_set = set(initial_seeds)
            
            for iteration in range(self.max_iterations):
                iteration_additions = set()
                
                for command in list(current_set):
                    # Adiciona comandos similares (preserva polaridade)
                    iteration_additions.update(self._get_similar_commands(command))
                    
                    # Adiciona comandos antônimos à polaridade oposta
                    if "Positive" in trait_polarity:
                        opposite_commands = self._get_antonym_commands(command)
                        # Estes iriam para o conjunto negativo (tratados separadamente)
                    
                    # Adiciona comandos derivados (preserva polaridade)
                    iteration_additions.update(self._get_derived_commands(command))
                    
                    # Adiciona membros da categoria (preserva polaridade)
                    iteration_additions.update(self._get_category_commands(command))
                
                new_commands = iteration_additions - current_set
                current_set.update(new_commands)
                
                logger.info(f"  Iteration {iteration + 1}: added {len(new_commands)} commands")
                
                if not new_commands:
                    break
            
            expanded_sets[trait_polarity] = current_set
            logger.info(f"Final {trait_polarity}: {len(current_set)} commands")
        
        return expanded_sets
    
    def _get_similar_commands(self, command: str) -> Set[str]:
        """Obtém comandos com funcionalidade similar."""
        similar = set()
        for base, similars in self.relations.get("similar", {}).items():
            if command.startswith(base):
                similar.update(similars)
            elif command in similars:
                similar.add(base)
        return similar
    
    def _get_antonym_commands(self, command: str) -> Set[str]:
        """Obtém comandos com funcionalidade oposta."""
        antonyms = set()
        for base, opposites in self.relations.get("antonym", {}).items():
            if command.startswith(base):
                antonyms.update(opposites)
            elif command in opposites:
                antonyms.add(base)
        return antonyms
    
    def _get_derived_commands(self, command: str) -> Set[str]:
        """Obtém comandos derivados da mesma ferramenta/família."""
        derived = set()
        for base, derivatives in self.relations.get("derived_from", {}).items():
            if command.startswith(base):
                derived.update(derivatives)
        return derived
    
    def _get_category_commands(self, command: str) -> Set[str]:
        """Obtém comandos da mesma categoria funcional."""
        category_commands = set()
        for category, commands in self.relations.get("also_see", {}).items():
            if command in commands:
                category_commands.update(commands)
        return category_commands - {command}

class RandomWalkRefiner:
    """
    Refina os escores de sentimento usando um algoritmo de Random Walk, 
    similar ao SentiWordNet 3.0.
    """
    
    def __init__(self, all_commands: List[str], relations: Dict, alpha: float = 0.85, iterations: int = 10):
        self.all_commands = all_commands
        self.cmd_to_idx = {cmd: i for i, cmd in enumerate(all_commands)}
        self.relations = relations
        self.alpha = alpha
        self.iterations = iterations
        self.transition_matrix = self._build_transition_matrix()
        
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Constrói a matriz de transição M onde M[i, j] é a probabilidade
        de ir do comando i para o comando j.
        """
        n = len(self.all_commands)
        M = np.zeros((n, n))
        
        logger.info(f"Construindo matriz de transição para {n} comandos...")
        
        # Inverte as relações para facilitar a busca
        rev_similar = {}
        for k, vals in self.relations.get("similar", {}).items():
            for v in vals:
                if v not in rev_similar:
                    rev_similar[v] = []
                rev_similar[v].append(k)
                
        rev_antonym = {}
        for k, vals in self.relations.get("antonym", {}).items():
            for v in vals:
                if v not in rev_antonym:
                    rev_antonym[v] = []
                rev_antonym[v].append(k)
                
        rev_derived = {}
        for k, vals in self.relations.get("derived_from", {}).items():
            for v in vals:
                if v not in rev_derived:
                    rev_derived[v] = []
                rev_derived[v].append(k)
        
        for i, cmd in enumerate(self.all_commands):
            neighbors = set()
            
            # Adiciona vizinhos de todas as relações diretas
            neighbors.update(self.relations.get("similar", {}).get(cmd, []))
            neighbors.update(self.relations.get("derived_from", {}).get(cmd, []))
            neighbors.update(self.relations.get("antonym", {}).get(cmd, []))
            
            # Adiciona comandos da mesma categoria funcional
            for cat, cmds in self.relations.get("also_see", {}).items():
                if cmd in cmds:
                    neighbors.update(c for c in cmds if c != cmd)

            # Adiciona relações invertidas
            neighbors.update(rev_similar.get(cmd, []))
            neighbors.update(rev_derived.get(cmd, []))
            neighbors.update(rev_antonym.get(cmd, []))

            # Preenche a matriz de transição
            valid_neighbors = []
            for neighbor in neighbors:
                if neighbor in self.cmd_to_idx:
                    j = self.cmd_to_idx[neighbor]
                    valid_neighbors.append(j)
            
            # Se não há vizinhos válidos, conecta a si mesmo (self-loop)
            if not valid_neighbors:
                M[i, i] = 1.0
            else:
                # Distribui probabilidade uniformemente entre vizinhos
                prob = 1.0 / len(valid_neighbors)
                for j in valid_neighbors:
                    M[i, j] = prob
        
        return M
    
    def refine_scores_multi_trait(self, initial_scores: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Executa o Random Walk para refinar os escores de múltiplos traços.
        Fórmula: S_t+1 = (1-alpha)*S0 + alpha * M^T * S_t
        
        Args:
            initial_scores: {command: {trait: {"positive": p, "negative": n, "objective": o}}}
        
        Returns:
            refined_scores: {command: {trait: {"positive": p, "negative": n, "objective": o}}}
        """
        n = len(self.all_commands)
        
        # Identifica todos os traços presentes
        all_traits = set()
        for cmd_scores in initial_scores.values():
            all_traits.update(cmd_scores.keys())
        all_traits = sorted(list(all_traits))
        
        logger.info(f"Refinando {len(all_traits)} traços: {all_traits}")
        
        M_T = self.transition_matrix.T
        refined_scores = {}
        
        # Processa cada traço separadamente
        for trait in all_traits:
            logger.info(f"Processando traço: {trait}")
            
            # Vetores de escores iniciais (S0) para este traço
            s0_pos = np.array([initial_scores.get(cmd, {}).get(trait, {}).get("positive", 0.0) for cmd in self.all_commands])
            s0_neg = np.array([initial_scores.get(cmd, {}).get(trait, {}).get("negative", 0.0) for cmd in self.all_commands])

            # Vetores de escores a serem iterados (St)
            s_pos = s0_pos.copy()
            s_neg = s0_neg.copy()

            logger.info(f"Iniciando Random Walk para {trait} com {self.iterations} iterações...")
            
            for i in range(self.iterations):
                # Aplica a fórmula do Random Walk
                s_pos_new = (1 - self.alpha) * s0_pos + self.alpha * (M_T @ s_pos)
                s_neg_new = (1 - self.alpha) * s0_neg + self.alpha * (M_T @ s_neg)
                
                # Calcula convergência
                conv_pos = np.mean(np.abs(s_pos_new - s_pos))
                conv_neg = np.mean(np.abs(s_neg_new - s_neg))
                
                s_pos = s_pos_new
                s_neg = s_neg_new
                
                if i % 2 == 0:  # Log a cada 2 iterações para reduzir verbosidade
                    logger.debug(f"  {trait} - Iteração {i+1}: conv_pos={conv_pos:.6f}, conv_neg={conv_neg:.6f}")

            # Armazena resultados refinados para este traço
            for i, cmd in enumerate(self.all_commands):
                if cmd not in refined_scores:
                    refined_scores[cmd] = {}
                
                pos = max(0.0, s_pos[i])
                neg = max(0.0, s_neg[i])
                obj = max(0.0, 1.0 - pos - neg)
                
                # Normaliza para somar 1.0
                total = pos + neg + obj
                if total > 0:
                    refined_scores[cmd][trait] = {
                        "positive": pos / total,
                        "negative": neg / total,
                        "objective": obj / total,
                    }
                else:
                    refined_scores[cmd][trait] = {"positive": 0.33, "negative": 0.33, "objective": 0.34}

        logger.info("Random Walk para todos os traços concluído!")
        return refined_scores
    
    def refine_scores(self, initial_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Versão legada para um único traço. Mantida para compatibilidade.
        """
        # Converte para formato multi-traço
        multi_trait_scores = {}
        for cmd, scores in initial_scores.items():
            multi_trait_scores[cmd] = {"default": scores}
        
        # Executa refinamento multi-traço
        refined_multi = self.refine_scores_multi_trait(multi_trait_scores)
        
        # Converte de volta para formato simples
        refined_simple = {}
        for cmd, trait_scores in refined_multi.items():
            refined_simple[cmd] = trait_scores["default"]
        
        return refined_simple

class VectorialClassifier:
    """
    Cria classificadores binários usando vetores TF-IDF de descrições de comandos.
    Segue a abordagem do SentiWordNet de usar dois classificadores binários por traço.
    """
    
    def __init__(self, gloss_extractor: CommandGlossExtractor):
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
            from sklearn.model_selection import StratifiedKFold
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
            "ls", "pwd", "cd", "echo", "cat", "less", "more", "head", "tail",
            "cp", "mv", "mkdir", "touch", "which", "whereis", "file",
            "date", "uptime", "who", "w", "id", "groups", "env"
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

def main():
    """Pipeline principal seguindo a metodologia SentiWordNet 3.0."""
    logger.info("=== Classificação Semi-Supervisionada de Comandos (Método SentiWordNet 3.0) ===")
    
    # Passo 1: Expande conjuntos semente
    expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS)
    expanded_sets = expander.expand_seeds()
    
    # --- NOVA ETAPA: PREPARAÇÃO PARA O RANDOM WALK (SentiWordNet 3.0) ---
    
    # Passo 2: Coleta todos os comandos únicos do universo conhecido PRIMEIRO
    logger.info("Coletando todos os comandos únicos para construção do grafo...")
    all_known_commands = set()
    
    # Adiciona comandos dos conjuntos expandidos
    for s in expanded_sets.values():
        all_known_commands.update(s)
    
    # Adiciona comandos das relações taxonômicas
    for rel_type in COMMAND_RELATIONS.values():
        for k, v_list in rel_type.items():
            all_known_commands.add(k)
            if isinstance(v_list, list):
                all_known_commands.update(v_list)
    
    # Adiciona alguns comandos de teste interessantes
    test_commands = ["rm -rf /tmp", "strace -f ls", "watch -n1 ps", "python exploit.py", 
                    "grep", "mount", "umount", "find", "chmod", "chown"]
    all_known_commands.update(test_commands)
    
    # Adiciona comandos objetivos/neutros (precisa ser criado temporariamente)
    temp_classifier = VectorialClassifier(CommandGlossExtractor())
    temp_labeled = set()
    all_known_commands.update(temp_classifier._get_objective_commands(temp_labeled, max_objective=30))
    
    all_known_commands = sorted(list(all_known_commands))
    logger.info(f"Total de comandos únicos para o grafo: {len(all_known_commands)}")
    
    # Passo 3: Prepara o vectorizer TF-IDF global com TODOS os comandos
    gloss_extractor = CommandGlossExtractor()
    classifier = VectorialClassifier(gloss_extractor)
    classifier.prepare_global_vectorizer(all_known_commands)
    
    # Passo 4: Treina classificadores vetoriais (SentiWordNet 1.0) usando vectorizer global
    trait_classifiers = classifier.train_trait_classifiers(expanded_sets)

    # Passo 5: Gera escores iniciais para TODOS os comandos (SentiWordNet 1.0)
    logger.info("Gerando escores iniciais para todos os comandos...")
    initial_scores = {}
    
    for cmd in all_known_commands:
        try:
            # Usa o classificador treinado para obter escores iniciais de todos os traços
            cmd_results = classifier.classify_command(cmd, trait_classifiers)
            initial_scores[cmd] = cmd_results
        except Exception as e:
            logger.warning(f"Erro ao classificar comando '{cmd}': {e}")
            # Escores padrão em caso de erro
            initial_scores[cmd] = {
                "Perfectionism": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Patience": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Creativity": {"positive": 0.33, "negative": 0.33, "objective": 0.34}
            }

    # Passo 6: Refina os escores com Random Walk (SentiWordNet 3.0)
    logger.info("Iniciando refinamento com Random Walk...")
    refiner = RandomWalkRefiner(
        all_known_commands, 
        COMMAND_RELATIONS, 
        alpha=0.70,  # Reduzido de 0.85 para 0.70 para mais refinamento
        iterations=25  # Aumentado de 15 para 25 iterações
    )
    final_scores = refiner.refine_scores_multi_trait(initial_scores)

    # Passo 7: Apresenta resultados comparativos
    print("\n" + "="*80)
    print("=== RESULTADOS DE CLASSIFICAÇÃO (SentiWordNet 3.0) ===")
    print("="*80)
    
    for cmd in test_commands:
        if cmd in final_scores:
            print(f"\nComando: {cmd}")
            print("-" * 60)
            
            for trait in ["Perfectionism", "Patience", "Creativity"]:
                if trait in final_scores[cmd]:
                    initial = initial_scores[cmd][trait]
                    refined = final_scores[cmd][trait]
                    
                    print(f"  {trait}:")
                    print(f"    Inicial:  pos={initial['positive']:.3f}, neg={initial['negative']:.3f}, obj={initial['objective']:.3f}")
                    print(f"    Refinado: pos={refined['positive']:.3f}, neg={refined['negative']:.3f}, obj={refined['objective']:.3f}")
                    
                    # Calcula diferença
                    diff_pos = refined['positive'] - initial['positive']
                    diff_neg = refined['negative'] - initial['negative']
                    print(f"    Mudança: pos={diff_pos:+.3f}, neg={diff_neg:+.3f}")
                    print()
        else:
            print(f"\nComando '{cmd}' não encontrado no grafo refinado.")
    
    # Passo 8: Salva resultados
    output = {
        "metadata": {
            "method": "SentiWordNet 3.0 with Random Walk",
            "total_commands": len(all_known_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha": refiner.alpha
        },
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "initial_scores": initial_scores,
        "refined_scores": final_scores,
        "test_commands": test_commands
    }
    
    # Salva resultados completos
    with open("semisupervised_results_v3.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    
    # Salva apenas escores refinados para uso rápido
    with open("refined_command_scores.json", "w") as f:
        json.dump(final_scores, f, indent=2, sort_keys=True)
    
    logger.info("Resultados salvos em 'semisupervised_results_v3.json' e 'refined_command_scores.json'")
    
    # Estatísticas finais
    print(f"\n" + "="*60)
    print("=== ESTATÍSTICAS FINAIS (OTIMIZADAS) ===")
    print(f"Comandos processados: {len(all_known_commands)}")
    print(f"Conjuntos semente expandidos:")
    for trait_polarity, commands in expanded_sets.items():
        print(f"  {trait_polarity}: {len(commands)} comandos")
    print(f"Iterações Random Walk: {refiner.iterations}")
    print(f"Fator de amortecimento (α): {refiner.alpha}")
    print(f"Traços analisados: {len(['Perfectionism', 'Patience', 'Creativity'])}")
    print(f"Vectorizer features: {len(classifier.vectorizer.vocabulary_)}")
    print(f"Validação cruzada: StratifiedKFold(5 splits, F1-weighted)")
    print("="*60)

if __name__ == "__main__":
    main() 