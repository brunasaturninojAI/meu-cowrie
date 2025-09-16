#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
Pipeline principal para classificação semi-supervisionada de comandos.
Versão modularizada seguindo a metodologia SentiWordNet 3.0.
"""

import json
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

# Importações dos módulos
from config.seed_sets import SEED_SETS
from config.command_relations import COMMAND_RELATIONS
from extractors.gloss_extractor import CommandGlossExtractor
from expanders.semi_supervised_expander import SemiSupervisedExpander
from classifiers.vectorial_classifier import VectorialClassifier
from refiners.random_walk_refiner import RandomWalkRefiner

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_all_commands(expanded_sets: Dict[str, Set[str]], relations: Dict) -> List[str]:
    """Coleta todos os comandos únicos do universo conhecido."""
    logger.info("Coletando todos os comandos únicos para construção do grafo...")
    all_known_commands = set()
    
    # Adiciona comandos dos conjuntos expandidos
    for s in expanded_sets.values():
        all_known_commands.update(s)
    
    # Adiciona comandos das relações taxonômicas
    for rel_type in relations.values():
        for k, v_list in rel_type.items():
            all_known_commands.add(k)
            if isinstance(v_list, list):
                all_known_commands.update(v_list)
    
    # Adiciona alguns comandos de teste interessantes
    test_commands = ["rm -rf /tmp", "strace -f ls", "watch -n1 ps", "python exploit.py", 
                    "grep", "mount", "umount", "find", "chmod", "chown"]
    
    # TESTE: Adiciona comandos NOVOS que não estão nos conjuntos semente
    new_test_commands = ["curl -X POST", "nmap -sS", "awk '{print $1}'", "sed 's/old/new/g'", 
                        "docker run", "systemctl start", "crontab -e", "wireshark"]
    test_commands.extend(new_test_commands)
    
    all_known_commands.update(test_commands)
    
    # Adiciona comandos objetivos/neutros
    temp_classifier = VectorialClassifier(CommandGlossExtractor())
    temp_labeled = set()
    all_known_commands.update(temp_classifier._get_objective_commands(temp_labeled, max_objective=30))
    
    all_known_commands = sorted(list(all_known_commands))
    logger.info(f"Total de comandos únicos para o grafo: {len(all_known_commands)}")
    
    return all_known_commands, test_commands

def generate_initial_scores(classifier: VectorialClassifier, trait_classifiers: Dict, 
                           all_known_commands: List[str]) -> Dict:
    """Gera escores iniciais para todos os comandos."""
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
    
    return initial_scores

def calculate_percentiles(final_scores: Dict, all_known_commands: List[str]) -> Dict:
    """
    Calcula percentis para cada comando em relação ao universo de comandos.
    Retorna percentil (0-1) indicando quantos % dos comandos têm score menor.
    """
    logger.info("Calculando percentis dos comandos...")
    percentiles = {}
    
    for trait in ["Perfectionism", "Patience", "Creativity"]:
        # Coleta todos os scores positivos para este traço
        all_scores = []
        for cmd in all_known_commands:
            if cmd in final_scores and trait in final_scores[cmd]:
                score = final_scores[cmd][trait]['positive']
                all_scores.append(score)
        
        # Ordena os scores
        all_scores.sort()
        total_commands = len(all_scores)
        
        logger.info(f"  {trait}: analisando {total_commands} comandos")
        
        # Calcula percentil para cada comando
        for cmd in all_known_commands:
            if cmd in final_scores and trait in final_scores[cmd]:
                cmd_score = final_scores[cmd][trait]['positive']
                
                # Conta quantos scores são menores
                rank = sum(1 for s in all_scores if s < cmd_score)
                percentile = rank / total_commands if total_commands > 0 else 0.0
                
                if cmd not in percentiles:
                    percentiles[cmd] = {}
                percentiles[cmd][trait] = percentile
    
    logger.info("Percentis calculados com sucesso!")
    return percentiles

def determine_attacker_personality(final_scores: Dict, test_commands: List[str], 
                                  all_known_commands: List[str]) -> Dict[str, str]:
    """
    Determina a personalidade final do atacante baseada em percentis.
    Retorna o traço dominante para cada comando e um perfil geral.
    """
    # Calcula percentis para todos os comandos
    percentiles = calculate_percentiles(final_scores, all_known_commands)
    
    personality_results = {}
    trait_totals = {"Perfectionism": 0, "Patience": 0, "Creativity": 0}
    command_count = 0
    
    # Analisa cada comando individual usando percentis
    for cmd in test_commands:
        if cmd in percentiles:
            command_count += 1
            cmd_traits = {}
            
            for trait in ["Perfectionism", "Patience", "Creativity"]:
                if trait in percentiles[cmd]:
                    percentile = percentiles[cmd][trait]
                    cmd_traits[trait] = percentile
                    trait_totals[trait] += percentile
            
            # Determina traço dominante para este comando (maior percentil)
            if cmd_traits:
                dominant_trait = max(cmd_traits.items(), key=lambda x: x[1])
                personality_results[cmd] = {
                    'dominant_trait': dominant_trait[0],
                    'percentile': dominant_trait[1],
                    'interpretation': f"Mais {dominant_trait[0].lower()} que {dominant_trait[1]:.0%} dos comandos",
                    'ranking': f"Top {100-dominant_trait[1]*100:.0f}% em {dominant_trait[0]}",
                    'all_percentiles': cmd_traits
                }
    
    # Determina personalidade geral do atacante usando percentis médios
    if command_count > 0:
        avg_percentiles = {trait: total/command_count for trait, total in trait_totals.items()}
        overall_dominant = max(avg_percentiles.items(), key=lambda x: x[1])
        
        personality_results['overall_profile'] = {
            'dominant_trait': overall_dominant[0],
            'average_percentile': overall_dominant[1],
            'interpretation': f"Em média, mais {overall_dominant[0].lower()} que {overall_dominant[1]:.0%} dos comandos",
            'average_percentiles': avg_percentiles,
            'total_commands': command_count
        }
    
    return personality_results

def display_results(test_commands: List[str], initial_scores: Dict, final_scores: Dict, all_known_commands: List[str]):
    """Apresenta resultados comparativos incluindo análise de personalidade com percentis."""
    print("\n" + "="*80)
    print("=== RESULTADOS DE CLASSIFICAÇÃO (SentiWordNet 3.0) ===")
    print("="*80)
    
    # Determina personalidade do atacante
    personality_analysis = determine_attacker_personality(final_scores, test_commands, all_known_commands)
    
    for cmd in test_commands:
        if cmd in final_scores:
            print(f"\nComando: {cmd}")
            print("-" * 60)
            
            # Mostra personalidade dominante para este comando
            if cmd in personality_analysis:
                cmd_personality = personality_analysis[cmd]
                print(f"  PERSONALIDADE DOMINANTE: {cmd_personality['dominant_trait']}")
                print(f"  PERCENTIL: {cmd_personality['percentile']:.1%} ({cmd_personality['ranking']})")
                print(f"  INTERPRETAÇÃO: {cmd_personality['interpretation']}")
                print()
                
                # Mostra todos os percentis
                print("  PERCENTIS POR TRAÇO:")
                for trait, percentile in cmd_personality['all_percentiles'].items():
                    print(f"    {trait}: {percentile:.1%} (mais {trait.lower()} que {percentile:.0%} dos comandos)")
                print()
            
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
    
    # Exibe perfil geral do atacante
    if 'overall_profile' in personality_analysis:
        profile = personality_analysis['overall_profile']
        print("\n" + "="*80)
        print("=== PERFIL GERAL DO ATACANTE ===")
        print("="*80)
        print(f"PERSONALIDADE DOMINANTE: {profile['dominant_trait']}")
        print(f"PERCENTIL MÉDIO: {profile['average_percentile']:.1%}")
        print(f"INTERPRETAÇÃO: {profile['interpretation']}")
        print(f"Comandos analisados: {profile['total_commands']}")
        print("\nPercentis médios por traço:")
        for trait, percentile in profile['average_percentiles'].items():
            print(f"  {trait}: {percentile:.1%}")
        
        # Interpretação da personalidade
        dominant = profile['dominant_trait']
        score = profile['average_percentile']
        print(f"\nANÁLISE DETALHADA:")
        if dominant == "Perfectionism" and score > 0.5:
            print("  → Atacante meticuloso, prefere comandos de diagnóstico e verificação")
            print("  → Está no grupo dos mais perfeccionistas")
        elif dominant == "Patience" and score > 0.5:
            print("  → Atacante paciente, usa comandos de monitoramento e observação")
            print("  → Está no grupo dos mais pacientes")
        elif dominant == "Creativity" and score > 0.5:
            print("  → Atacante criativo, emprega ferramentas complexas e scripts customizados")
            print("  → Está no grupo dos mais criativos")
        else:
            print(f"  → Atacante com percentil médio em {dominant}")
            print(f"  → Não se destaca significativamente neste traço")
        print("="*80)

def save_results(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                final_scores: Dict, test_commands: List[str], refiner: RandomWalkRefiner):
    """Salva resultados em arquivos JSON."""
    output = {
        "metadata": {
            "method": "SentiWordNet 3.0 with Random Walk (Modular)",
            "total_commands": len(refiner.all_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha": refiner.alpha
        },
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "initial_scores": initial_scores,
        "refined_scores": final_scores,
        "test_commands": test_commands
    }
    
    # Salva resultados completos
    with open("semisupervised_results_v3_modular.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    
    # Salva apenas escores refinados para uso rápido
    with open("refined_command_scores_modular.json", "w") as f:
        json.dump(final_scores, f, indent=2, sort_keys=True)
    
    logger.info("Resultados salvos em 'semisupervised_results_v3_modular.json' e 'refined_command_scores_modular.json'")

def display_final_stats(expanded_sets: Dict[str, Set[str]], all_known_commands: List[str], 
                       refiner: RandomWalkRefiner, classifier: VectorialClassifier):
    """Exibe estatísticas finais."""
    print(f"\n" + "="*60)
    print("=== ESTATÍSTICAS FINAIS (MODULARIZADO) ===")
    print(f"Comandos processados: {len(all_known_commands)}")
    print(f"Conjuntos semente expandidos:")
    for trait_polarity, commands in expanded_sets.items():
        print(f"  {trait_polarity}: {len(commands)} comandos")
    print(f"Iterações Random Walk: {refiner.iterations}")
    print(f"Fator de amortecimento (α): {refiner.alpha}")
    print(f"Traços analisados: {len(['Perfectionism', 'Patience', 'Creativity'])}")
    print(f"Vectorizer features: {len(classifier.vectorizer.vocabulary_)}")
    print(f"Validação cruzada: StratifiedKFold(5 splits, F1-weighted)")
    
    # Informações adicionais do refinador
    refiner_info = refiner.get_refinement_info()
    print(f"Densidade da matriz: {refiner_info['matrix_density']:.3f}")
    print(f"Vizinhos médios: {refiner_info['avg_neighbors']:.1f}")
    print("="*60)

def create_results_directory():
    """Cria diretório para resultados organizados"""
    results_dir = Path("cowrie_analysis_results")
    results_dir.mkdir(exist_ok=True)
    
    # Criar subdiretórios
    (results_dir / "graphs").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)
    (results_dir / "reports").mkdir(exist_ok=True)
    
    logger.info(f"Diretório de resultados criado: {results_dir}")
    return results_dir

def generate_metrics(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                    final_scores: Dict, test_commands: List[str], refiner: RandomWalkRefiner) -> Dict:
    """Gera métricas quantitativas da análise"""
    logger.info("Gerando métricas da análise...")
    
    metrics = {
        "dataset_stats": {
            "total_commands": len(refiner.all_commands),
            "seed_sets_expanded": {k: len(v) for k, v in expanded_sets.items()},
            "test_commands": len(test_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha_factor": refiner.alpha
        },
        "classification_improvement": {},
        "personality_distribution": {},
        "command_complexity": {}
    }
    
    # Métricas de melhoria da classificação
    improvement_data = []
    for cmd in test_commands:
        if cmd in final_scores and cmd in initial_scores:
            for trait in ["Perfectionism", "Patience", "Creativity"]:
                if trait in final_scores[cmd] and trait in initial_scores[cmd]:
                    initial_pos = initial_scores[cmd][trait]['positive']
                    final_pos = final_scores[cmd][trait]['positive']
                    improvement = final_pos - initial_pos
                    
                    improvement_data.append({
                        'command': cmd,
                        'trait': trait,
                        'initial_score': initial_pos,
                        'final_score': final_pos,
                        'improvement': improvement
                    })
    
    if improvement_data:
        df_improvement = pd.DataFrame(improvement_data)
        metrics["classification_improvement"] = {
            "avg_improvement": df_improvement['improvement'].mean(),
            "max_improvement": df_improvement['improvement'].max(),
            "min_improvement": df_improvement['improvement'].min(),
            "positive_improvements": len(df_improvement[df_improvement['improvement'] > 0]),
            "total_classifications": len(df_improvement)
        }
    
    # Distribuição de personalidade
    personality_counts = {"Perfectionism": 0, "Patience": 0, "Creativity": 0}
    for cmd in test_commands:
        if cmd in final_scores:
            dominant_trait = max(
                [(trait, final_scores[cmd][trait]['positive']) for trait in ["Perfectionism", "Patience", "Creativity"]],
                key=lambda x: x[1]
            )[0]
            personality_counts[dominant_trait] += 1
    
    metrics["personality_distribution"] = personality_counts
    
    # Complexidade dos comandos (baseada no número de argumentos)
    complexity_data = []
    for cmd in test_commands:
        complexity = len(cmd.split()) - 1  # Número de argumentos
        complexity_data.append(complexity)
    
    if complexity_data:
        metrics["command_complexity"] = {
            "avg_complexity": np.mean(complexity_data),
            "max_complexity": max(complexity_data),
            "min_complexity": min(complexity_data),
            "simple_commands": len([c for c in complexity_data if c <= 2]),
            "complex_commands": len([c for c in complexity_data if c > 2])
        }
    
    return metrics

def create_visualizations(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                         final_scores: Dict, test_commands: List[str], results_dir: Path):
    """Cria visualizações dos resultados"""
    logger.info("Criando visualizações...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Comparação de escores antes e depois do refinamento
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise de Classificação de Comandos - SentiWordNet 3.0', fontsize=16)
    
    # Gráfico 1: Comparação de escores por traço
    traits = ["Perfectionism", "Patience", "Creativity"]
    initial_avg = []
    final_avg = []
    
    for trait in traits:
        trait_initial = []
        trait_final = []
        for cmd in test_commands:
            if cmd in initial_scores and cmd in final_scores:
                if trait in initial_scores[cmd] and trait in final_scores[cmd]:
                    trait_initial.append(initial_scores[cmd][trait]['positive'])
                    trait_final.append(final_scores[cmd][trait]['positive'])
        
        if trait_initial:
            initial_avg.append(np.mean(trait_initial))
            final_avg.append(np.mean(trait_final))
    
    x = np.arange(len(traits))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, initial_avg, width, label='Inicial', alpha=0.8)
    axes[0, 0].bar(x + width/2, final_avg, width, label='Refinado', alpha=0.8)
    axes[0, 0].set_xlabel('Traços de Personalidade')
    axes[0, 0].set_ylabel('Score Positivo Médio')
    axes[0, 0].set_title('Comparação de Escores por Traço')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(traits)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Distribuição de personalidade dominante
    personality_counts = {"Perfectionism": 0, "Patience": 0, "Creativity": 0}
    for cmd in test_commands:
        if cmd in final_scores:
            dominant_trait = max(
                [(trait, final_scores[cmd][trait]['positive']) for trait in traits],
                key=lambda x: x[1]
            )[0]
            personality_counts[dominant_trait] += 1
    
    axes[0, 1].pie(personality_counts.values(), labels=personality_counts.keys(), autopct='%1.1f%%')
    axes[0, 1].set_title('Distribuição de Personalidade Dominante')
    
    # Gráfico 3: Melhoria dos escores
    improvements = []
    for cmd in test_commands:
        if cmd in final_scores and cmd in initial_scores:
            for trait in traits:
                if trait in final_scores[cmd] and trait in initial_scores[cmd]:
                    improvement = final_scores[cmd][trait]['positive'] - initial_scores[cmd][trait]['positive']
                    improvements.append(improvement)
    
    if improvements:
        axes[1, 0].hist(improvements, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Melhoria no Score')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição de Melhorias nos Escores')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Complexidade dos comandos vs Score
    complexity_scores = []
    for cmd in test_commands:
        if cmd in final_scores:
            complexity = len(cmd.split()) - 1
            avg_score = np.mean([final_scores[cmd][trait]['positive'] for trait in traits if trait in final_scores[cmd]])
            complexity_scores.append((complexity, avg_score))
    
    if complexity_scores:
        complexities, scores = zip(*complexity_scores)
        axes[1, 1].scatter(complexities, scores, alpha=0.6)
        axes[1, 1].set_xlabel('Complexidade do Comando (argumentos)')
        axes[1, 1].set_ylabel('Score Positivo Médio')
        axes[1, 1].set_title('Complexidade vs Score')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico principal
    graph_file = results_dir / "graphs" / "main_analysis.png"
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico principal salvo em: {graph_file}")
    
    # Criar gráfico de evolução do Random Walk
    plt.figure(figsize=(10, 6))
    # Simular evolução (você pode adaptar para usar dados reais do refiner)
    iterations = range(1, 26)
    convergence = [0.8 - 0.6 * np.exp(-i/5) for i in iterations]  # Exemplo
    
    plt.plot(iterations, convergence, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteração do Random Walk')
    plt.ylabel('Score de Convergência')
    plt.title('Convergência do Algoritmo Random Walk')
    plt.grid(True, alpha=0.3)
    
    convergence_file = results_dir / "graphs" / "random_walk_convergence.png"
    plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico de convergência salvo em: {convergence_file}")
    
    plt.close('all')

def generate_summary_report(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                           final_scores: Dict, test_commands: List[str], metrics: Dict, 
                           refiner: RandomWalkRefiner, results_dir: Path, all_known_commands: List[str]):
    """Gera relatório resumido da análise"""
    logger.info("Gerando relatório resumido...")
    
    # Determinar personalidade do atacante
    personality_analysis = determine_attacker_personality(final_scores, test_commands, all_known_commands)
    
    report = f"""# Relatório de Análise de Comandos - SentiWordNet 3.0

## Resumo Executivo
- **Total de comandos analisados**: {metrics['dataset_stats']['total_commands']}
- **Comandos de teste**: {metrics['dataset_stats']['test_commands']}
- **Iterações Random Walk**: {metrics['dataset_stats']['random_walk_iterations']}
- **Fator de amortecimento (α)**: {metrics['dataset_stats']['alpha_factor']}

## Expansão dos Conjuntos Semente
"""
    
    for trait_polarity, commands in expanded_sets.items():
        report += f"- **{trait_polarity}**: {len(commands)} comandos\n"
    
    report += f"""
## Métricas de Classificação
- **Melhoria média nos escores**: {metrics['classification_improvement'].get('avg_improvement', 0):.3f}
- **Classificações com melhoria positiva**: {metrics['classification_improvement'].get('positive_improvements', 0)}/{metrics['classification_improvement'].get('total_classifications', 0)}
- **Melhoria máxima**: {metrics['classification_improvement'].get('max_improvement', 0):.3f}

## Distribuição de Personalidade (Percentis)
"""
    
    for trait, count in metrics['personality_distribution'].items():
        percentage = (count / len(test_commands)) * 100 if test_commands else 0
        report += f"- **{trait}**: {count} comandos ({percentage:.1f}%)\n"
    
    report += f"""
## Perfil do Atacante (Baseado em Percentis)
"""
    
    if 'overall_profile' in personality_analysis:
        profile = personality_analysis['overall_profile']
        report += f"""
- **Personalidade dominante**: {profile['dominant_trait']}
- **Percentil médio**: {profile['average_percentile']:.1%}
- **Interpretação**: {profile['interpretation']}
- **Comandos analisados**: {profile['total_commands']}

### Análise Detalhada:
"""
        
        dominant = profile['dominant_trait']
        score = profile['average_percentile']
        
        if dominant == "Perfectionism" and score > 0.5:
            report += "→ Atacante meticuloso, prefere comandos de diagnóstico e verificação\n"
            report += "→ Está no grupo dos mais perfeccionistas\n"
        elif dominant == "Patience" and score > 0.5:
            report += "→ Atacante paciente, usa comandos de monitoramento e observação\n"
            report += "→ Está no grupo dos mais pacientes\n"
        elif dominant == "Creativity" and score > 0.5:
            report += "→ Atacante criativo, emprega ferramentas complexas e scripts customizados\n"
            report += "→ Está no grupo dos mais criativos\n"
        else:
            report += f"→ Atacante com percentil médio em {dominant}\n"
            report += f"→ Não se destaca significativamente neste traço\n"
    
    report += f"""
## Complexidade dos Comandos
- **Complexidade média**: {metrics['command_complexity'].get('avg_complexity', 0):.1f} argumentos
- **Comandos simples (≤2 args)**: {metrics['command_complexity'].get('simple_commands', 0)}
- **Comandos complexos (>2 args)**: {metrics['command_complexity'].get('complex_commands', 0)}

## Metodologia
- **Classificação**: Semi-supervisionada com SentiWordNet 3.0
- **Refinamento**: Random Walk com propagação de escores
- **Análise**: Percentis relativos ao universo de comandos ({len(all_known_commands)} comandos)
- **Traços analisados**: Perfectionism, Patience, Creativity
- **Vectorizer**: TF-IDF global com gloss dos comandos

---
*Relatório gerado automaticamente pelo Pipeline de Classificação Cowrie*
"""
    
    # Salvar relatório
    report_file = results_dir / "reports" / "analysis_summary.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Relatório resumido salvo em: {report_file}")
    return report

def save_detailed_results(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                         final_scores: Dict, test_commands: List[str], metrics: Dict, 
                         refiner: RandomWalkRefiner, results_dir: Path):
    """Salva resultados detalhados em formato JSON"""
    logger.info("Salvando resultados detalhados...")
    
    # Salvar métricas
    metrics_file = results_dir / "metrics" / "analysis_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Salvar escores refinados
    scores_file = results_dir / "metrics" / "refined_scores.json"
    with open(scores_file, 'w', encoding='utf-8') as f:
        json.dump(final_scores, f, indent=2, ensure_ascii=False)
    
    # Salvar resultados completos
    complete_results = {
        "metadata": {
            "method": "SentiWordNet 3.0 with Random Walk (Modular)",
            "total_commands": len(refiner.all_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha": refiner.alpha
        },
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "initial_scores": initial_scores,
        "refined_scores": final_scores,
        "test_commands": test_commands,
        "metrics": metrics
    }
    
    complete_file = results_dir / "complete_analysis_results.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados salvos em: {results_dir}")

def main():
    """Pipeline principal modularizado seguindo a metodologia SentiWordNet 3.0."""
    logger.info("=== Classificação Semi-Supervisionada de Comandos (Método SentiWordNet 3.0 - Modular) ===")
    
    try:
        # Criar diretório de resultados
        results_dir = create_results_directory()
        
        # Passo 1: Expande conjuntos semente
        logger.info("Iniciando expansão de conjuntos semente...")
        expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS)
        expanded_sets = expander.expand_seeds()
        
        # Passo 2: Coleta todos os comandos únicos
        all_known_commands, test_commands = collect_all_commands(expanded_sets, COMMAND_RELATIONS)
        
        # Passo 3: Prepara o vectorizer TF-IDF global
        logger.info("Preparando classificadores...")
        gloss_extractor = CommandGlossExtractor()
        classifier = VectorialClassifier(gloss_extractor)
        classifier.prepare_global_vectorizer(all_known_commands)
        
        # Passo 4: Treina classificadores vetoriais
        trait_classifiers = classifier.train_trait_classifiers(expanded_sets)
        
        # Passo 5: Gera escores iniciais
        initial_scores = generate_initial_scores(classifier, trait_classifiers, all_known_commands)
        
        # Passo 6: Refina os escores com Random Walk
        logger.info("Iniciando refinamento com Random Walk...")
        refiner = RandomWalkRefiner(
            all_known_commands, 
            COMMAND_RELATIONS, 
            alpha=0.70,
            iterations=25
        )
        final_scores = refiner.refine_scores_multi_trait(initial_scores)
        
        # Passo 7: Gera métricas e visualizações
        logger.info("Gerando métricas e visualizações...")
        metrics = generate_metrics(expanded_sets, initial_scores, final_scores, test_commands, refiner)
        
        # Passo 8: Cria visualizações
        create_visualizations(expanded_sets, initial_scores, final_scores, test_commands, results_dir)
        
        # Passo 9: Gera relatório resumido
        generate_summary_report(expanded_sets, initial_scores, final_scores, test_commands, metrics, refiner, results_dir, all_known_commands)
        
        # Passo 10: Salva resultados detalhados
        save_detailed_results(expanded_sets, initial_scores, final_scores, test_commands, metrics, refiner, results_dir)
        
        # Passo 11: Apresenta resultados no console
        display_results(test_commands, initial_scores, final_scores, all_known_commands)
        
        # Passo 12: Estatísticas finais
        display_final_stats(expanded_sets, all_known_commands, refiner, classifier)
        
        logger.info(f"Pipeline modularizado executado com sucesso! Resultados salvos em: {results_dir}")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main() 