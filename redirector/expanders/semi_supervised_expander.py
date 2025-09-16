#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
semi_supervised_expander.py
Expansão semi-supervisionada seguindo a metodologia SentiWordNet.
Expande conjuntos semente usando relações taxonômicas iterativamente.
"""

from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

class SemiSupervisedExpander:
    """
    Expansão semi-supervisionada seguindo a metodologia SentiWordNet.
    Expande conjuntos semente usando relações taxonômicas iterativamente.
    """
    
    def __init__(self, seed_sets: Dict[str, List[str]], relations: Dict[str, Dict], max_iterations: int = 3):
        self.seed_sets = seed_sets
        self.relations = relations
        self.max_iterations = max_iterations
        
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
                        _ = self._get_antonym_commands(command)  # Antônimos coletados, mas não utilizados aqui
                    
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
    
    def get_expansion_stats(self, expanded_sets: Dict[str, Set[str]]) -> Dict[str, int]:
        """Retorna estatísticas da expansão."""
        stats = {}
        for trait_polarity, commands in expanded_sets.items():
            original_count = len(self.seed_sets[trait_polarity])
            expanded_count = len(commands)
            stats[trait_polarity] = {
                "original": original_count,
                "expanded": expanded_count,
                "growth": expanded_count - original_count,
                "growth_rate": (expanded_count / original_count) if original_count > 0 else 0
            }
        return stats 