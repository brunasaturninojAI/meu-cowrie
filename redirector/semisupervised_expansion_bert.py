#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
semisupervised_expansion_bert.py
Versão melhorada usando BERT para embeddings contextuais de comandos.

Melhorias sobre TF-IDF:
1. Embeddings contextuais de alta qualidade
2. Melhor captura de similaridade semântica
3. Zero-shot learning para comandos não vistos
4. Representações densas mais eficientes
5. Melhor performance em classificação
"""

import json
import subprocess
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import logging

# Importações do BERT
BERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("AVISO: transformers não instalado. Use: pip install transformers torch")
except Exception as e:
    BERT_AVAILABLE = False
    print(f"AVISO: Erro ao importar BERT: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONJUNTOS SEMENTE (iguais ao original) ---
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

# --- RELAÇÕES TAXONÔMICAS (iguais ao original) ---
COMMAND_RELATIONS = {
    "similar": {
        "ps": ["top", "htop"],
        "ls": ["dir", "ll"],
        "cat": ["more", "less"],
        "grep": ["egrep", "fgrep"],
        "vi": ["vim", "nano"],
        "rm": ["rmdir", "unlink"],
    },
    "antonym": {
        "start": ["stop", "kill"],
        "mount": ["umount"],
        "compress": ["decompress", "uncompress"],
        "encrypt": ["decrypt"],
        "connect": ["disconnect"],
    },
    "derived_from": {
        "gcc": ["g++", "gdb"],
        "git": ["gitk", "git-log"],
        "docker": ["docker-compose"],
        "ssh": ["scp", "sftp"],
    },
    "also_see": {
        "network_analysis": ["netstat", "ss", "lsof", "tcpdump"],
        "process_monitoring": ["ps", "top", "htop", "pstree"],
        "file_operations": ["ls", "find", "locate", "which"],
        "system_info": ["uname", "lscpu", "lsblk", "df"],
    }
}

class BERTCommandEncoder:
    """
    Encoder de comandos usando BERT para gerar embeddings contextuais.
    Substitui TF-IDF por representações semânticas mais ricas.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", cache_dir: str = "./bert_cache"):
        if not BERT_AVAILABLE:
            raise ImportError("BERT não disponível. Instale: pip install transformers torch")
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Carregando modelo BERT: {model_name}")
        logger.info(f"Usando device: {self.device}")
        
        # Carrega tokenizer e modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Cache para embeddings computados
        self.embedding_cache = {}
        
    def encode_command_description(self, command: str, description: str) -> np.ndarray:
        """
        Gera embedding contextual para comando + descrição.
        
        Args:
            command: Nome do comando (ex: "rm -rf")
            description: Descrição/gloss do comando
            
        Returns:
            np.ndarray: Embedding de 768 dimensões (DistilBERT)
        """
        cache_key = f"{command}|{description}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Formata texto para o BERT: "[CLS] comando: descrição [SEP]"
        text = f"command {command}: {description}"
        
        # Tokeniza
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Gera embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Usa pooled output (representação do [CLS] token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def encode_commands_batch(self, command_descriptions: List[Tuple[str, str]], batch_size: int = 8) -> np.ndarray:
        """
        Gera embeddings para múltiplos comandos em batch (mais eficiente).
        
        Args:
            command_descriptions: Lista de (comando, descrição)
            batch_size: Tamanho do lote para processamento
            
        Returns:
            np.ndarray: Matrix de embeddings [n_commands, embedding_dim]
        """
        # Verifica cache primeiro
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, (cmd, desc) in enumerate(command_descriptions):
            cache_key = f"{cmd}|{desc}"
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                text = f"command {cmd}: {desc}"
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                embeddings.append(None)  # placeholder
        
        # Encode batch dos não-cachados em lotes menores
        if texts_to_encode:
            logger.info(f"Encodando {len(texts_to_encode)} comandos novos em lotes de {batch_size}...")
            
            all_batch_embeddings = []
            
            # Processa em lotes para evitar problemas de memória
            for i in range(0, len(texts_to_encode), batch_size):
                batch_texts = texts_to_encode[i:i + batch_size]
                
                # Tokeniza lote atual
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Gera embeddings para este lote
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_batch_embeddings.append(batch_embeddings)
            
            # Concatena todos os lotes
            batch_embeddings = np.vstack(all_batch_embeddings)
            
            # Atualiza cache e lista de embeddings
            for j, idx in enumerate(indices_to_encode):
                cmd, desc = command_descriptions[idx]
                cache_key = f"{cmd}|{desc}"
                
                embedding = batch_embeddings[j]
                self.embedding_cache[cache_key] = embedding
                embeddings[idx] = embedding
        
        return np.array(embeddings)
    
    def find_similar_commands(self, target_command: str, target_description: str, 
                             command_pool: List[Tuple[str, str]], 
                             top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Encontra comandos similares usando similaridade de embeddings BERT.
        
        Args:
            target_command: Comando de referência
            target_description: Descrição do comando de referência  
            command_pool: Pool de (comando, descrição) para buscar
            top_k: Número máximo de similares a retornar
            threshold: Threshold mínimo de similaridade
            
        Returns:
            Lista de (comando, score_similaridade) ordenada por similaridade
        """
        # Embedding do comando alvo
        target_embedding = self.encode_command_description(target_command, target_description)
        
        # Embeddings do pool (usando batch menor para evitar problemas de memória)
        pool_embeddings = self.encode_commands_batch(command_pool, batch_size=4)
        
        # Calcula similaridades
        similarities = cosine_similarity([target_embedding], pool_embeddings)[0]
        
        # Filtra e ordena
        similar_commands = []
        for i, (cmd, _) in enumerate(command_pool):
            sim = similarities[i]
            if sim >= threshold and cmd != target_command:  # Evita self-match
                similar_commands.append((cmd, sim))
        
        # Ordena por similaridade decrescente
        similar_commands.sort(key=lambda x: x[1], reverse=True)
        
        return similar_commands[:top_k]

class CommandGlossExtractor:
    """Extrai 'glossas' (descrições) das páginas man (igual ao original)."""
    
    def __init__(self):
        self.cache = {}
    
    def get_command_gloss(self, command: str) -> str:
        """Extrai descrição da página man (igual ao original)."""
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

class BERTSemiSupervisedExpander:
    """
    Versão melhorada do SemiSupervisedExpander usando BERT para descoberta de similaridades.
    """
    
    def __init__(self, seed_sets: Dict[str, List[str]], relations: Dict[str, Dict], 
                 bert_encoder: BERTCommandEncoder, max_iterations: int = 3, 
                 similarity_threshold: float = 0.75):
        self.seed_sets = seed_sets
        self.relations = relations
        self.bert_encoder = bert_encoder
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.gloss_extractor = CommandGlossExtractor()
        
    def expand_seeds_with_bert(self, command_universe: List[str]) -> Dict[str, Set[str]]:
        """
        Expande conjuntos semente usando tanto relações taxonômicas quanto BERT.
        """
        expanded_sets = {}
        
        # Prepara pool de comandos com descrições
        logger.info("Preparando descrições para todos os comandos...")
        command_descriptions = [
            (cmd, self.gloss_extractor.get_command_gloss(cmd)) 
            for cmd in command_universe
        ]
        
        for trait_polarity, initial_seeds in self.seed_sets.items():
            logger.info(f"Expanding {trait_polarity}: {len(initial_seeds)} initial seeds")
            
            current_set = set(initial_seeds)
            
            for iteration in range(self.max_iterations):
                iteration_additions = set()
                
                for command in list(current_set):
                    # 1. Adiciona comandos via relações taxonômicas (método original)
                    iteration_additions.update(self._get_similar_commands(command))
                    iteration_additions.update(self._get_derived_commands(command))
                    iteration_additions.update(self._get_category_commands(command))
                    
                    # 2. NOVO: Adiciona comandos via similaridade BERT
                    cmd_description = self.gloss_extractor.get_command_gloss(command)
                    bert_similar = self.bert_encoder.find_similar_commands(
                        command, cmd_description, command_descriptions,
                        top_k=3, threshold=self.similarity_threshold
                    )
                    
                    # Adiciona apenas os comandos (sem os scores)
                    for similar_cmd, score in bert_similar:
                        iteration_additions.add(similar_cmd)
                        logger.debug(f"  BERT similar to '{command}': '{similar_cmd}' (score: {score:.3f})")
                
                new_commands = iteration_additions - current_set
                current_set.update(new_commands)
                
                logger.info(f"  Iteration {iteration + 1}: added {len(new_commands)} commands")
                
                if not new_commands:
                    break
            
            expanded_sets[trait_polarity] = current_set
            logger.info(f"Final {trait_polarity}: {len(current_set)} commands")
        
        return expanded_sets
    
    def _get_similar_commands(self, command: str) -> Set[str]:
        """Obtém comandos com funcionalidade similar (método original)."""
        similar = set()
        for base, similars in self.relations.get("similar", {}).items():
            if command.startswith(base):
                similar.update(similars)
            elif command in similars:
                similar.add(base)
        return similar
    
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

class BERTVectorialClassifier:
    """
    Classificador usando embeddings BERT em vez de TF-IDF.
    Mantém a lógica de classificadores binários do SentiWordNet.
    """
    
    def __init__(self, bert_encoder: BERTCommandEncoder, gloss_extractor: CommandGlossExtractor):
        self.bert_encoder = bert_encoder
        self.gloss_extractor = gloss_extractor
        self.classifiers = {}
        self.command_embeddings = {}
        
    def train_trait_classifiers(self, expanded_sets: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        Treina classificadores usando embeddings BERT.
        """
        trait_classifiers = {}
        
        # Agrupa por traço (ignorando polaridade)
        traits = set()
        for trait_polarity in expanded_sets.keys():
            trait = trait_polarity.split('_')[0]  # "Perfectionism_Positive" -> "Perfectionism"
            traits.add(trait)
        
        for trait in traits:
            logger.info(f"Training BERT classifiers for {trait}")
            
            pos_key = f"{trait}_Positive"
            neg_key = f"{trait}_Negative"
            
            if pos_key not in expanded_sets or neg_key not in expanded_sets:
                logger.warning(f"Missing data for {trait}, skipping")
                continue
            
            positive_commands = list(expanded_sets[pos_key])
            negative_commands = list(expanded_sets[neg_key])
            
            # Cria conjunto objetivo (comandos neutros)
            all_commands = set(positive_commands + negative_commands)
            objective_commands = self._get_objective_commands(all_commands)
            
            # Prepara dados: (comando, descrição) para cada conjunto
            pos_data = [(cmd, self.gloss_extractor.get_command_gloss(cmd)) for cmd in positive_commands]
            neg_data = [(cmd, self.gloss_extractor.get_command_gloss(cmd)) for cmd in negative_commands]
            obj_data = [(cmd, self.gloss_extractor.get_command_gloss(cmd)) for cmd in objective_commands]
            
            # Gera embeddings BERT em batch (usando batch menor para estabilidade)
            all_data = pos_data + neg_data + obj_data
            X = self.bert_encoder.encode_commands_batch(all_data, batch_size=4)
            
            # Prepara rótulos para classificadores binários
            # Classificador 1: Positivo vs (Negativo + Objetivo)
            y_positive = ([1] * len(pos_data) + 
                         [0] * len(neg_data) + 
                         [0] * len(obj_data))
            
            # Classificador 2: Negativo vs (Positivo + Objetivo)  
            y_negative = ([0] * len(pos_data) + 
                         [1] * len(neg_data) + 
                         [0] * len(obj_data))
            
            # Treina classificadores
            clf_positive = SVC(probability=True, random_state=42, kernel='rbf')
            clf_negative = SVC(probability=True, random_state=42, kernel='rbf')
            
            clf_positive.fit(X, y_positive)
            clf_negative.fit(X, y_negative)
            
            # Avalia performance
            pos_score = cross_val_score(clf_positive, X, y_positive, cv=3).mean()
            neg_score = cross_val_score(clf_negative, X, y_negative, cv=3).mean()
            
            logger.info(f"  {trait} BERT positive classifier CV score: {pos_score:.3f}")
            logger.info(f"  {trait} BERT negative classifier CV score: {neg_score:.3f}")
            
            trait_classifiers[trait] = (clf_positive, clf_negative)
        
        return trait_classifiers
    
    def _get_objective_commands(self, labeled_commands: Set[str], max_objective: int = 50) -> List[str]:
        """Obtém comandos objetivos (neutros)."""
        common_commands = [
            "ls", "pwd", "cd", "echo", "cat", "less", "more", "head", "tail",
            "cp", "mv", "mkdir", "touch", "which", "whereis", "file",
            "date", "uptime", "who", "w", "id", "groups", "env"
        ]
        
        objective = [cmd for cmd in common_commands if cmd not in labeled_commands]
        return objective[:max_objective]
    
    def classify_command(self, command: str, trait_classifiers: Dict[str, Tuple]) -> Dict[str, Dict[str, float]]:
        """
        Classifica um comando usando embeddings BERT.
        """
        # Gera embedding para o comando
        gloss = self.gloss_extractor.get_command_gloss(command)
        X = self.bert_encoder.encode_command_description(command, gloss).reshape(1, -1)
        
        results = {}
        
        for trait, (clf_pos, clf_neg) in trait_classifiers.items():
            # Obtém probabilidades
            pos_proba = clf_pos.predict_proba(X)[0][1]  
            neg_proba = clf_neg.predict_proba(X)[0][1]  
            
            # Aplica lógica de decisão do SentiWordNet
            if pos_proba > 0.5 and neg_proba <= 0.5:
                positive_score = pos_proba
                negative_score = 0.0
                objective_score = 1.0 - positive_score
            elif pos_proba <= 0.5 and neg_proba > 0.5:
                positive_score = 0.0
                negative_score = neg_proba
                objective_score = 1.0 - negative_score
            else:
                positive_score = max(0.0, pos_proba - 0.5) if pos_proba > neg_proba else 0.0
                negative_score = max(0.0, neg_proba - 0.5) if neg_proba > pos_proba else 0.0
                objective_score = 1.0 - positive_score - negative_score
            
            # Normaliza
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

# Reutiliza RandomWalkRefiner do arquivo original (sem modificações)
class RandomWalkRefiner:
    """Refina os escores usando Random Walk (igual ao original)."""
    
    def __init__(self, all_commands: List[str], relations: Dict, alpha: float = 0.85, iterations: int = 10):
        self.all_commands = all_commands
        self.cmd_to_idx = {cmd: i for i, cmd in enumerate(all_commands)}
        self.relations = relations
        self.alpha = alpha
        self.iterations = iterations
        self.transition_matrix = self._build_transition_matrix()
        
    def _build_transition_matrix(self) -> np.ndarray:
        """Constrói a matriz de transição (igual ao original)."""
        n = len(self.all_commands)
        M = np.zeros((n, n))
        
        logger.info(f"Construindo matriz de transição para {n} comandos...")
        
        # [Implementação igual ao original - código omitido por brevidade]
        # ... (copiar implementação do original)
        
        return M
    
    def refine_scores_multi_trait(self, initial_scores: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Executa Random Walk para refinar escores (igual ao original)."""
        # [Implementação igual ao original - código omitido por brevidade]
        # ... (copiar implementação do original)
        pass

def main():
    """Pipeline principal usando BERT."""
    logger.info("=== Classificação Semi-Supervisionada com BERT ===")
    
    if not BERT_AVAILABLE:
        logger.error("BERT não disponível! Instale: pip install transformers torch")
        return
    
    # Passo 1: Inicializa encoder BERT
    logger.info("Inicializando BERT encoder...")
    bert_encoder = BERTCommandEncoder()
    
    # Passo 2: Coleta comandos do universo
    logger.info("Coletando comandos do universo...")
    all_known_commands = set()
    
    # Adiciona sementes
    for s in SEED_SETS.values():
        all_known_commands.update(s)
    
    # Adiciona relações
    for rel_type in COMMAND_RELATIONS.values():
        for k, v_list in rel_type.items():
            all_known_commands.add(k)
            if isinstance(v_list, list):
                all_known_commands.update(v_list)
    
    # Adiciona comandos de teste
    test_commands = ["rm -rf /tmp", "strace -f ls", "watch -n1 ps", "python exploit.py", 
                    "grep", "mount", "umount", "find", "chmod", "chown"]
    all_known_commands.update(test_commands)
    
    all_known_commands = sorted(list(all_known_commands))
    logger.info(f"Total de comandos: {len(all_known_commands)}")
    
    # Passo 3: Expande conjuntos semente com BERT
    bert_available = True
    try:
        expander = BERTSemiSupervisedExpander(
            SEED_SETS, COMMAND_RELATIONS, bert_encoder,
            similarity_threshold=0.75
        )
        expanded_sets = expander.expand_seeds_with_bert(all_known_commands)
        logger.info("Expansão BERT concluída com sucesso!")
    except Exception as e:
        logger.error(f"Erro na expansão BERT: {e}")
        logger.info("Fazendo fallback para expansão básica...")
        # Fallback para expansão básica sem BERT
        from semisupervised_expansion import SemiSupervisedExpander as BasicExpander
        expander = BasicExpander(SEED_SETS, COMMAND_RELATIONS)
        expanded_sets = expander.expand_seeds()
        bert_available = False
    
    # Passo 4: Treina classificadores BERT
    try:
        gloss_extractor = CommandGlossExtractor()
        classifier = BERTVectorialClassifier(bert_encoder, gloss_extractor)
        trait_classifiers = classifier.train_trait_classifiers(expanded_sets)
        logger.info("Classificadores BERT treinados com sucesso!")
    except Exception as e:
        logger.error(f"Erro no treinamento BERT: {e}")
        logger.info("Fazendo fallback para classificadores básicos...")
        # Fallback para classificadores básicos
        from semisupervised_expansion import VectorialClassifier as BasicClassifier
        basic_classifier = BasicClassifier(gloss_extractor)
        # Prepara vectorizer global primeiro
        all_commands_list = list(all_known_commands)
        basic_classifier.prepare_global_vectorizer(all_commands_list)
        trait_classifiers = basic_classifier.train_trait_classifiers(expanded_sets)
        classifier = basic_classifier  # Para usar no passo seguinte
        bert_available = False
    
    # Passo 5: Gera escores iniciais
    logger.info("Gerando escores iniciais...")
    initial_scores = {}
    
    for cmd in all_known_commands:
        try:
            cmd_results = classifier.classify_command(cmd, trait_classifiers)
            initial_scores[cmd] = cmd_results
        except Exception as e:
            logger.warning(f"Erro ao classificar '{cmd}': {e}")
            initial_scores[cmd] = {
                "Perfectionism": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Patience": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Creativity": {"positive": 0.33, "negative": 0.33, "objective": 0.34}
            }
    
    # Passo 6: Refina com Random Walk (opcional - usar implementação original)
    # ... (pode ser adicionado aqui)
    
    # Passo 7: Apresenta resultados
    print("\n" + "="*80)
    method_title = "BERT" if bert_available else "TF-IDF (Fallback)"
    print(f"=== RESULTADOS DE CLASSIFICAÇÃO ({method_title}) ===")
    print("="*80)
    
    for cmd in test_commands:
        if cmd in initial_scores:
            print(f"\nComando: {cmd}")
            print("-" * 60)
            
            for trait in ["Perfectionism", "Patience", "Creativity"]:
                if trait in initial_scores[cmd]:
                    scores = initial_scores[cmd][trait]
                    print(f"  {trait}:")
                    print(f"    pos={scores['positive']:.3f}, neg={scores['negative']:.3f}, obj={scores['objective']:.3f}")
    
    # Passo 8: Salva resultados
    output = {
        "metadata": {
            "method": "BERT + SentiWordNet" if bert_available else "TF-IDF Fallback",
            "model": getattr(bert_encoder, 'model_name', 'fallback') if bert_available and 'bert_encoder' in locals() else "tfidf",
            "total_commands": len(all_known_commands),
            "similarity_threshold": getattr(expander, 'similarity_threshold', 0.75) if bert_available and hasattr(expander, 'similarity_threshold') else 0.0
        },
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "bert_scores": initial_scores,
        "test_commands": test_commands
    }
    
    output_filename = "bert_semisupervised_results.json" if bert_available else "tfidf_fallback_results.json"
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    
    logger.info(f"Resultados salvos em '{output_filename}'")

if __name__ == "__main__":
    main() 