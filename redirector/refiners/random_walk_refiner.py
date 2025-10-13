#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
random_walk_refiner.py
Refina os escores de sentimento usando um algoritmo de Random Walk, 
similar ao SentiWordNet 3.0.
"""

from typing import Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RandomWalkRefiner:
    """
    Refina os escores de sentimento usando um algoritmo de Random Walk, 
    similar ao SentiWordNet 3.0.
    """
    
    def __init__(self, all_commands: List[str], relations: Dict, alpha: float = 0.60, iterations: int = 20, tolerance: float = 1e-4, patience: int = 5):
        self.all_commands = all_commands
        self.cmd_to_idx = {cmd: i for i, cmd in enumerate(all_commands)}
        self.relations = relations
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance
        self.patience = patience
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
        
        # Pesos diferenciados por tipo de relação
        w_sim = 1.0   # Similar: peso máximo
        w_der = 0.7   # Derived: reduzido de 0.8
        w_cat = 0.5   # Category: reduzido de 0.6
        w_ant = 0.05  # Antônimos: muito reduzido (era 0.15) para minimizar contaminação cruzada

        for i, cmd in enumerate(self.all_commands):
            neighbors = set()
            weighted_edges = {}
            
            # Adiciona vizinhos de todas as relações diretas
            for nb in self.relations.get("similar", {}).get(cmd, []):
                neighbors.add(nb)
                weighted_edges[nb] = max(weighted_edges.get(nb, 0.0), w_sim)
            for nb in self.relations.get("derived_from", {}).get(cmd, []):
                neighbors.add(nb)
                weighted_edges[nb] = max(weighted_edges.get(nb, 0.0), w_der)
            for nb in self.relations.get("antonym", {}).get(cmd, []):
                neighbors.add(nb)
                weighted_edges[nb] = max(weighted_edges.get(nb, 0.0), w_ant)
            
            # Adiciona comandos da mesma categoria funcional
            for cat, cmds in self.relations.get("also_see", {}).items():
                if cmd in cmds:
                    for c in cmds:
                        if c == cmd:
                            continue
                        neighbors.add(c)
                        weighted_edges[c] = max(weighted_edges.get(c, 0.0), w_cat)

            # Adiciona relações invertidas
            for nb in rev_similar.get(cmd, []):
                neighbors.add(nb)
                weighted_edges[nb] = max(weighted_edges.get(nb, 0.0), w_sim)
            for nb in rev_derived.get(cmd, []):
                neighbors.add(nb)
                weighted_edges[nb] = max(weighted_edges.get(nb, 0.0), w_der)
            for nb in rev_antonym.get(cmd, []):
                neighbors.add(nb)
                weighted_edges[nb] = max(weighted_edges.get(nb, 0.0), w_ant)

            # Preenche a matriz de transição
            # Constrói lista de vizinhos válidos e seus pesos
            valid_neighbors = []
            valid_weights = []
            for neighbor, weight in weighted_edges.items():
                if neighbor in self.cmd_to_idx:
                    j = self.cmd_to_idx[neighbor]
                    valid_neighbors.append(j)
                    valid_weights.append(weight)

            # Se não há vizinhos válidos, conecta a si mesmo (self-loop)
            if not valid_neighbors:
                M[i, i] = 1.0
            else:
                # Normaliza pesos para formar distribuição de probabilidade
                weights = np.array(valid_weights, dtype=float)
                # Pequeno epsilon para estabilidade
                denom = weights.sum()
                if denom <= 0:
                    probs = np.full(len(valid_neighbors), 1.0 / len(valid_neighbors))
                else:
                    probs = weights / denom
                for idx, j in enumerate(valid_neighbors):
                    M[i, j] = probs[idx]
        
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
            
            stable_steps = 0
            for i in range(self.iterations):
                # Aplica a fórmula do Random Walk
                s_pos_new = (1 - self.alpha) * s0_pos + self.alpha * (M_T @ s_pos)
                s_neg_new = (1 - self.alpha) * s0_neg + self.alpha * (M_T @ s_neg)
                
                # Calcula convergência
                conv_pos = np.mean(np.abs(s_pos_new - s_pos))
                conv_neg = np.mean(np.abs(s_neg_new - s_neg))
                
                s_pos = s_pos_new
                s_neg = s_neg_new
                
                if i % 5 == 0:  # Log a cada 5 iterações para reduzir verbosidade
                    logger.debug(f"  {trait} - Iteração {i+1}: conv_pos={conv_pos:.6f}, conv_neg={conv_neg:.6f}")

                # Early stopping quando convergir por 'patience' passos
                if conv_pos < self.tolerance and conv_neg < self.tolerance:
                    stable_steps += 1
                    if stable_steps >= self.patience:
                        logger.debug(f"  {trait} - Early stopping na iteração {i+1} (convergência estável)")
                        break
                else:
                    stable_steps = 0

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
    
    def get_refinement_info(self) -> Dict:
        """Retorna informações sobre o refinamento."""
        return {
            "total_commands": len(self.all_commands),
            "alpha": self.alpha,
            "iterations": self.iterations,
            "matrix_density": np.count_nonzero(self.transition_matrix) / (len(self.all_commands) ** 2),
            "avg_neighbors": np.mean(np.sum(self.transition_matrix > 0, axis=1))
        } 