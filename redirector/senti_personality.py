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

# --- CONJUNTOS SEMENTE (curados manualmente como os 47 positivos + 58 negativos do SentiWordNet) ---
SEED_SETS = {
    "Perfectionism_Positive": [
        "dmesg", "fsck", "diff", "strace", "lsof", "rpm -V", 
        "badblocks", "e2fsck", "getconf", "lscpu", "ethtool"
    ],
    "Perfectionism_Negative": [
        "rm -rf", "kill -9", "iptables -F", "rpm --nodeeps", 
        "nohup", "shutdown -h now"
    ],
    "Patience_Positive": [
        "watch", "tcpdump", "tail -f", "man", "apropos", 
        "sleep", "wait"
    ],
    "Patience_Negative": [
        "pkill", "kill -9", "ctrl+c", "timeout"
    ],
    "Creativity_Positive": [
        "gcc", "python", "git", "curl", "wget", "nc", "ssh",
        "vim", "emacs", "make"
    ],
    "Creativity_Negative": [
        "cat", "ls", "pwd", "cd"  # comandos básicos/repetitivos
    ]
}

# --- RELAÇÕES TAXONÔMICAS (adaptadas das relações do WordNet) ---
COMMAND_RELATIONS = {
    # Funcionalidade similar (como "similar-to" do WordNet)
    "similar": {
        "ps": ["top", "htop"],
        "ls": ["dir", "ll"],
        "cat": ["more", "less"],
        "grep": ["egrep", "fgrep"],
        "vi": ["vim", "nano"],
        "rm": ["rmdir", "unlink"],
    },
    
    # Funcionalidade oposta (como "antonym" do WordNet)
    "antonym": {
        "start": ["stop", "kill"],
        "mount": ["umount"],
        "compress": ["decompress", "uncompress"],
        "encrypt": ["decrypt"],
        "connect": ["disconnect"],
    },
    
    # Relações de ferramentas (como "derived-from" do WordNet)
    "derived_from": {
        "gcc": ["g++", "gdb"],
        "git": ["gitk", "git-log"],
        "docker": ["docker-compose"],
        "ssh": ["scp", "sftp"],
    },
    
    # Categorias funcionais (como "also-see" do WordNet)
    "also_see": {
        "network_analysis": ["netstat", "ss", "lsof", "tcpdump"],
        "process_monitoring": ["ps", "top", "htop", "pstree"],
        "file_operations": ["ls", "find", "locate", "which"],
        "system_info": ["uname", "lscpu", "lsblk", "df"],
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

class VectorialClassifier:
    """
    Cria classificadores binários usando vetores TF-IDF de descrições de comandos.
    Segue a abordagem do SentiWordNet de usar dois classificadores binários por traço.
    """
    
    def __init__(self, gloss_extractor: CommandGlossExtractor):
        self.gloss_extractor = gloss_extractor
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.classifiers = {}
        
    def train_trait_classifiers(self, expanded_sets: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        Treina classificadores binários para cada traço.
        Retorna dicionário de pares (classificador_positivo, classificador_negativo).
        """
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
            
            # Treina vetorizador TF-IDF
            X = self.vectorizer.fit_transform(all_glosses)
            
            # Prepara rótulos para classificadores binários
            # Classificador 1: Positivo vs (Negativo + Objetivo)
            y_positive = ([1] * len(pos_glosses) + 
                         [0] * len(neg_glosses) + 
                         [0] * len(obj_glosses))
            
            # Classificador 2: Negativo vs (Positivo + Objetivo)  
            y_negative = ([0] * len(pos_glosses) + 
                         [1] * len(neg_glosses) + 
                         [0] * len(obj_glosses))
            
            # Treina classificadores binários
            clf_positive = SVC(probability=True, random_state=42)
            clf_negative = SVC(probability=True, random_state=42)
            
            clf_positive.fit(X, y_positive)
            clf_negative.fit(X, y_negative)
            
            # Avalia
            pos_score = cross_val_score(clf_positive, X, y_positive, cv=3).mean()
            neg_score = cross_val_score(clf_negative, X, y_negative, cv=3).mean()
            
            logger.info(f"  {trait} positive classifier CV score: {pos_score:.3f}")
            logger.info(f"  {trait} negative classifier CV score: {neg_score:.3f}")
            
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
            
            # Aplica lógica de decisão do SentiWordNet
            if pos_proba > 0.5 and neg_proba <= 0.5:
                # Positivo
                positive_score = pos_proba
                negative_score = 0.0
                objective_score = 1.0 - positive_score
            elif pos_proba <= 0.5 and neg_proba > 0.5:
                # Negativo  
                positive_score = 0.0
                negative_score = neg_proba
                objective_score = 1.0 - negative_score
            else:
                # Objetivo (ambos baixos ou ambos altos)
                positive_score = max(0.0, pos_proba - 0.5) if pos_proba > neg_proba else 0.0
                negative_score = max(0.0, neg_proba - 0.5) if neg_proba > pos_proba else 0.0
                objective_score = 1.0 - positive_score - negative_score
            
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
    """Pipeline principal seguindo a metodologia SentiWordNet."""
    logger.info("=== Classificação Semi-Supervisionada de Comandos (Método SentiWordNet) ===")
    
    # Passo 1: Expande conjuntos semente
    expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS)
    expanded_sets = expander.expand_seeds()
    
    # Passo 2: Treina classificadores vetoriais
    gloss_extractor = CommandGlossExtractor()
    classifier = VectorialClassifier(gloss_extractor)
    trait_classifiers = classifier.train_trait_classifiers(expanded_sets)
    
    # Passo 3: Testa em comandos de exemplo
    test_commands = ["rm -rf /tmp", "strace -f ls", "watch -n1 ps", "python exploit.py"]
    
    print("\n=== Classification Results ===")
    for cmd in test_commands:
        results = classifier.classify_command(cmd, trait_classifiers)
        print(f"\nCommand: {cmd}")
        for trait, scores in results.items():
            print(f"  {trait}:")
            for polarity, score in scores.items():
                print(f"    {polarity}: {score:.3f}")
    
    # Passo 4: Salva modelo
    output = {
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "test_results": {cmd: classifier.classify_command(cmd, trait_classifiers) 
                        for cmd in test_commands}
    }
    
    with open("semisupervised_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info("Results saved to semisupervised_results.json")

if __name__ == "__main__":
    main() 