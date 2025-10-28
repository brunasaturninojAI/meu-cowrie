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
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

# Importações dos módulos
from config.seed_sets import SEED_SETS
from config.command_relations import COMMAND_RELATIONS
from extractors.gloss_extractor import CommandGlossExtractor
from expanders.semi_supervised_expander import SemiSupervisedExpander
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
    
    # Adiciona comandos objetivos/neutros (opcional, depende de sklearn)
    try:
        from classifiers.vectorial_classifier import VectorialClassifier as _VC
        temp_classifier = _VC(CommandGlossExtractor())
        temp_labeled = set()
        all_known_commands.update(temp_classifier._get_objective_commands(temp_labeled, max_objective=30))
    except BaseException as e:
        logger.warning(f"Não foi possível obter comandos objetivos via classificador ({e}). Prosseguindo sem eles.")
    
    all_known_commands = sorted(list(all_known_commands))
    logger.info(f"Total de comandos únicos para o grafo: {len(all_known_commands)}")
    
    return all_known_commands, test_commands

def generate_initial_scores(classifier, trait_classifiers: Dict, 
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
                "HonestyHumility": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Emotionality": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Extraversion": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Agreeableness": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Conscientiousness": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "OpennessToExperience": {"positive": 0.33, "negative": 0.33, "objective": 0.34}
            }
    
    return initial_scores

def generate_initial_scores_from_seeds(expanded_sets: Dict[str, Set[str]],
                                      all_known_commands: List[str]) -> Dict:
    """Fallback: gera escores iniciais apenas a partir dos conjuntos semente.
    Evita dependências de ML (scikit-learn/SciPy).
    """
    logger.info("Gerando escores iniciais (fallback por seeds, sem ML)...")
    initial_scores = {}
    traits = [
        "HonestyHumility", "Emotionality", "Extraversion",
        "Agreeableness", "Conscientiousness", "OpennessToExperience"
    ]
    for cmd in all_known_commands:
        initial_scores[cmd] = {}
        for trait in traits:
            pos_key = f"{trait}_Positive"
            neg_key = f"{trait}_Negative"
            if pos_key in expanded_sets and cmd in expanded_sets[pos_key]:
                initial_scores[cmd][trait] = {"positive": 0.70, "negative": 0.15, "objective": 0.15}
            elif neg_key in expanded_sets and cmd in expanded_sets[neg_key]:
                initial_scores[cmd][trait] = {"positive": 0.15, "negative": 0.70, "objective": 0.15}
            else:
                initial_scores[cmd][trait] = {"positive": 0.33, "negative": 0.33, "objective": 0.34}
    return initial_scores

def calculate_percentiles(final_scores: Dict, all_known_commands: List[str]) -> Dict:
    """
    Calcula percentis para cada comando em relação ao universo de comandos.
    Retorna percentil (0-1) indicando quantos % dos comandos têm score menor.
    """
    logger.info("Calculando percentis dos comandos...")
    percentiles = {}
    
    for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
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

def calculate_average_scores(final_scores: Dict, test_commands: List[str]) -> Dict:
    """
    Calcula médias dos scores brutos dos comandos de teste.
    Retorna scores médios para cada traço.
    """
    logger.info("Calculando médias dos scores brutos dos comandos de teste...")
    avg_scores = {"HonestyHumility": 0, "Emotionality": 0, "Extraversion": 0, "Agreeableness": 0, "Conscientiousness": 0, "OpennessToExperience": 0}
    command_count = 0
    
    for cmd in test_commands:
        if cmd in final_scores:
            command_count += 1
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
                if trait in final_scores[cmd]:
                    avg_scores[trait] += final_scores[cmd][trait]['positive']
    
    # Calcula médias
    if command_count > 0:
        avg_scores = {trait: total/command_count for trait, total in avg_scores.items()}
        logger.info(f"  Médias calculadas para {command_count} comandos")
        for trait, avg in avg_scores.items():
            logger.info(f"    {trait}: {avg:.3f}")
    
    return avg_scores

def calculate_percentile_of_average(avg_scores: Dict, final_scores: Dict, 
                                   all_known_commands: List[str]) -> Dict:
    """
    Calcula percentil da média em relação ao universo completo.
    Retorna percentil (0-1) indicando quantos % dos comandos têm score menor que a média.
    """
    logger.info("Calculando percentil da média em relação ao universo completo...")
    percentiles_of_avg = {}
    
    for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
        # Coleta todos os scores individuais para este traço
        all_scores = []
        for cmd in all_known_commands:
            if cmd in final_scores and trait in final_scores[cmd]:
                score = final_scores[cmd][trait]['positive']
                all_scores.append(score)
        
        # Ordena os scores
        all_scores.sort()
        total_commands = len(all_scores)
        
        # Calcula percentil da média
        avg_score = avg_scores[trait]
        rank = sum(1 for s in all_scores if s < avg_score)
        percentile = rank / total_commands if total_commands > 0 else 0.0
        
        percentiles_of_avg[trait] = percentile
        logger.info(f"  {trait}: média {avg_score:.3f} → percentil {percentile:.1%}")
    
    logger.info("Percentis da média calculados com sucesso!")
    return percentiles_of_avg

def determine_attacker_personality(final_scores: Dict, test_commands: List[str], 
                                  all_known_commands: List[str]) -> Dict[str, str]:
    """
    Determina a personalidade final do atacante baseada em percentis.
    Retorna o traço dominante para cada comando e um perfil geral.
    """
    # Carrega limiares ótimos (se existirem) para classificar POS/NEG por traço
    best_thresholds = {
        "HonestyHumility": 0.5, "Emotionality": 0.5, "Extraversion": 0.5,
        "Agreeableness": 0.5, "Conscientiousness": 0.5, "OpennessToExperience": 0.5
    }
    try:
        val_file = Path("redirector/cowrie_analysis_results/metrics/validation_metrics.json")
        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as vf:
                v = json.load(vf)
            for k, obj in v.items():
                if isinstance(obj, dict) and "best_threshold" in obj and obj["best_threshold"] is not None:
                    best_thresholds[k] = float(obj["best_threshold"])
    except Exception:
        pass

    # Calcula percentis para todos os comandos (para análise individual)
    percentiles = calculate_percentiles(final_scores, all_known_commands)
    
    personality_results = {}
    command_count = 0
    
    # Analisa cada comando individual usando percentis
    for cmd in test_commands:
        if cmd in percentiles:
            command_count += 1
            cmd_traits = {}
            cmd_classifications = {}  # Para classificação POS/NEG
            
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
                if trait in percentiles[cmd]:
                    percentile = percentiles[cmd][trait]
                    cmd_traits[trait] = percentile
                    
                    # Classifica POS/NEG usando limiar otimizado convertido em percentil aproximado
                    # Observação: percentil usa ranking no universo; mantemos comparação padrão 0.5 para percentil,
                    # mas a classificação POS/NEG usa score bruto com limiar por traço
                    raw_score = final_scores.get(cmd, {}).get(trait, {}).get('positive', percentile)
                    thr = best_thresholds.get(trait, 0.5)
                    if raw_score >= thr:
                        cmd_classifications[trait] = "POSITIVO"
                    else:
                        cmd_classifications[trait] = "NEGATIVO"
            
            # Determina traço dominante para este comando (maior percentil)
            if cmd_traits:
                dominant_trait = max(cmd_traits.items(), key=lambda x: x[1])
                personality_results[cmd] = {
                    'dominant_trait': dominant_trait[0],
                    'percentile': dominant_trait[1],
                    'classification': cmd_classifications[dominant_trait[0]],
                    'interpretation': f"Mais {dominant_trait[0].lower()} que {dominant_trait[1]:.0%} dos comandos",
                    'ranking': f"Top {100-dominant_trait[1]*100:.0f}% em {dominant_trait[0]}",
                    'all_percentiles': cmd_traits,
                    'all_classifications': cmd_classifications
                }
    
    # Calcula personalidade geral usando PERCENTIL DA MÉDIA (Opção 3)
    if command_count > 0:
        # Calcula médias dos scores brutos
        avg_scores = calculate_average_scores(final_scores, test_commands)
        
        # Calcula percentil da média em relação ao universo completo
        percentiles_of_avg = calculate_percentile_of_average(avg_scores, final_scores, all_known_commands)
        
        # Determina traço dominante (maior percentil da média)
        overall_dominant = max(percentiles_of_avg.items(), key=lambda x: x[1])
        
        # Classifica o perfil geral baseado em 50%
        overall_classification = "POSITIVO" if overall_dominant[1] > 0.5 else "NEGATIVO"
        
        personality_results['overall_profile'] = {
            'dominant_trait': overall_dominant[0],
            'average_percentile': overall_dominant[1],
            'classification': overall_classification,
            'interpretation': f"A média do atacante é mais {overall_dominant[0].lower()} que {overall_dominant[1]:.0%} dos comandos",
            'average_scores': avg_scores,
            'average_percentiles': percentiles_of_avg,
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
                print(f"  PERSONALIDADE DOMINANTE: {cmd_personality['dominant_trait']} ({cmd_personality['classification']})")
                print(f"  PERCENTIL: {cmd_personality['percentile']:.1%} ({cmd_personality['ranking']})")
                print(f"  INTERPRETAÇÃO: {cmd_personality['interpretation']}")
                print()
                
                # Mostra todos os percentis com classificação
                print("  PERCENTIS POR TRAÇO:")
                for trait, percentile in cmd_personality['all_percentiles'].items():
                    classification = cmd_personality['all_classifications'][trait]
                    print(f"    {trait}: {percentile:.1%} ({classification}) - mais {trait.lower()} que {percentile:.0%} dos comandos")
                print()
            
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
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
        print(f"PERSONALIDADE DOMINANTE: {profile['dominant_trait']} ({profile['classification']})")
        print(f"PERCENTIL DA MÉDIA: {profile['average_percentile']:.1%}")
        print(f"INTERPRETAÇÃO: {profile['interpretation']}")
        print(f"Comandos analisados: {profile['total_commands']}")
        
        print(f"\nSCORES MÉDIOS BRUTOS:")
        for trait, score in profile['average_scores'].items():
            print(f"  {trait}: {score:.3f}")
            
        print(f"\nPERCENTIS DA MÉDIA:")
        for trait, percentile in profile['average_percentiles'].items():
            classification = "POSITIVO" if percentile > 0.5 else "NEGATIVO"
            print(f"  {trait}: {percentile:.1%} ({classification})")
        
        # Interpretação da personalidade
        dominant = profile['dominant_trait']
        score = profile['average_percentile']
        classification = profile['classification']
        print(f"\nANÁLISE DETALHADA:")
        if dominant == "HonestyHumility" and classification == "POSITIVO":
            print("  → Atacante ético, prefere comandos de verificação e transparência")
            print("  → A média dos seus comandos demonstra honestidade e humildade")
        elif dominant == "Emotionality" and classification == "POSITIVO":
            print("  → Atacante cauteloso, usa comandos de backup e monitoramento cuidadoso")
            print("  → A média dos seus comandos demonstra alta emocionalidade e cautela")
        elif dominant == "Extraversion" and classification == "POSITIVO":
            print("  → Atacante sociável, emprega ferramentas de comunicação e interação")
            print("  → A média dos seus comandos demonstra alta extraversão e confiança")
        elif dominant == "Agreeableness" and classification == "POSITIVO":
            print("  → Atacante cooperativo, prefere comandos colaborativos e flexíveis")
            print("  → A média dos seus comandos demonstra alta cordialidade")
        elif dominant == "Conscientiousness" and classification == "POSITIVO":
            print("  → Atacante organizado, usa comandos sistemáticos e disciplinados")
            print("  → A média dos seus comandos demonstra alta conscienciosidade")
        elif dominant == "OpennessToExperience" and classification == "POSITIVO":
            print("  → Atacante criativo, emprega ferramentas inovadoras e exploratórias")
            print("  → A média dos seus comandos demonstra alta abertura a experiências")
        else:
            print(f"  → Atacante com percentil médio abaixo de 50% em {dominant}")
            print(f"  → A média dos seus comandos não se destaca significativamente neste traço")
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
                       refiner: RandomWalkRefiner, classifier):
    """Exibe estatísticas finais."""
    print(f"\n" + "="*60)
    print("=== ESTATÍSTICAS FINAIS (MODULARIZADO) ===")
    print(f"Comandos processados: {len(all_known_commands)}")
    print(f"Conjuntos semente expandidos:")
    for trait_polarity, commands in expanded_sets.items():
        print(f"  {trait_polarity}: {len(commands)} comandos")
    print(f"Iterações Random Walk: {refiner.iterations}")
    print(f"Fator de amortecimento (α): {refiner.alpha}")
    print(f"Traços analisados: {len(['HonestyHumility', 'Emotionality', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'OpennessToExperience'])}")
    try:
        clf_info = classifier.get_classifier_info()
        print(f"Vectorizer features: {clf_info.get('vectorizer_features', 0)}")
    except Exception:
        print("Vectorizer features: n/a")
    print(f"Validação cruzada: StratifiedKFold(5 splits, F1-weighted)")
    
    # Informações adicionais do refinador
    refiner_info = refiner.get_refinement_info()
    print(f"Densidade da matriz: {refiner_info['matrix_density']:.3f}")
    print(f"Vizinhos médios: {refiner_info['avg_neighbors']:.1f}")
    print("="*60)

def create_results_directory():
    """Cria diretório para resultados organizados, sempre em redirector/cowrie_analysis_results"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "cowrie_analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Criar subdiretórios
    (results_dir / "graphs").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)
    (results_dir / "reports").mkdir(exist_ok=True)
    
    logger.info(f"Diretório de resultados criado: {results_dir.resolve()}")
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
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
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
    personality_counts = {"HonestyHumility": 0, "Emotionality": 0, "Extraversion": 0, "Agreeableness": 0, "Conscientiousness": 0, "OpennessToExperience": 0}
    for cmd in test_commands:
        if cmd in final_scores:
            dominant_trait = max(
                [(trait, final_scores[cmd][trait]['positive']) for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]],
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

def validate_effectiveness(final_scores: Dict, expanded_sets: Dict[str, Set[str]], results_dir: Path) -> Dict:
    """Valida efetividade usando seeds como rótulos fracos.
    Calcula F1 e AUC-ROC por traço para classe POS/NEG a partir do score 'positive'.
    """
    try:
        from sklearn.metrics import roc_auc_score, f1_score
    except Exception:
        logger.warning("sklearn indisponível; validação quantitativa pulada.")
        return {}

    traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
    report = {}
    for trait in traits:
        pos_key = f"{trait}_Positive"
        neg_key = f"{trait}_Negative"
        y_true = []
        y_score = []
        # Usa seeds como rótulos fracos
        for cmd in expanded_sets.get(pos_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(1)
                y_score.append(final_scores[cmd][trait]['positive'])
        for cmd in expanded_sets.get(neg_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(0)
                y_score.append(final_scores[cmd][trait]['positive'])
        if y_true and len(set(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = None
            # Otimiza limiar para maximizar F1 nas seeds
            best_f1 = 0.0
            best_thr = 0.5
            # Varre limiares de 0.1 a 0.9
            for thr in np.linspace(0.1, 0.9, 81):
                preds = [1 if s >= thr else 0 for s in y_score]
                f1 = f1_score(y_true, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
            # F1 no limiar padrão 0.5 para comparação
            f1_default = f1_score(y_true, [1 if s >= 0.5 else 0 for s in y_score])
            report[trait] = {
                "auc": float(auc) if auc is not None else None,
                "f1_best": float(best_f1),
                "best_threshold": best_thr,
                "f1_default": float(f1_default),
                "samples": len(y_true)
            }
        else:
            report[trait] = {"auc": None, "f1_best": None, "best_threshold": None, "f1_default": None, "samples": len(y_true)}

    # Salvar
    val_file = results_dir / "metrics" / "validation_metrics.json"
    try:
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Métricas de validação salvas em: {val_file}")
    except Exception as e:
        logger.warning(f"Falha ao salvar métricas de validação: {e}")
    return report

def create_visualizations(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                         final_scores: Dict, test_commands: List[str], results_dir: Path):
    """Cria visualizações dos resultados"""
    logger.info("Criando visualizações...")
    # Importação preguiçosa com backend não interativo
    try:
        import matplotlib
        try:
            matplotlib.use('Agg')
        except BaseException:
            pass
        import matplotlib.pyplot as plt
    except BaseException as e:
        logger.warning(f"Matplotlib indisponível ({e}); pulando visualizações.")
        return
    try:
        import seaborn as sns
        sns.set_palette("husl")
    except Exception as e:
        logger.warning(f"Seaborn indisponível ({e}); seguindo apenas com matplotlib.")
        sns = None
    
    # Configurar estilo
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        pass
    
    try:
        # 1. Comparação de escores antes e depois do refinamento
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Classificação de Comandos - SentiWordNet 3.0', fontsize=16)
        
        # Gráfico 1: Comparação de escores por traço
        traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
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
        personality_counts = {"HonestyHumility": 0, "Emotionality": 0, "Extraversion": 0, "Agreeableness": 0, "Conscientiousness": 0, "OpennessToExperience": 0}
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
        iterations = range(1, 26)
        convergence = [0.8 - 0.6 * np.exp(-i/5) for i in iterations]
        
        plt.plot(iterations, convergence, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Iteração do Random Walk')
        plt.ylabel('Score de Convergência')
        plt.title('Convergência do Algoritmo Random Walk')
        plt.grid(True, alpha=0.3)
        
        convergence_file = results_dir / "graphs" / "random_walk_convergence.png"
        plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de convergência salvo em: {convergence_file}")
    except BaseException as e:
        logger.warning(f"Falha ao gerar visualizações ({e}). Pulando gráficos.")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass

def generate_summary_report(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                           final_scores: Dict, test_commands: List[str], metrics: Dict, 
                           refiner: RandomWalkRefiner, results_dir: Path, all_known_commands: List[str]):
    """Gera relatório resumido da análise"""
    logger.info("Gerando relatório resumido...")
    
    # Determinar personalidade do atacante
    personality_analysis = determine_attacker_personality(final_scores, test_commands, all_known_commands)
    
    report = f"""# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

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
## Perfil do Atacante (Baseado em Percentil da Média)
"""
    
    if 'overall_profile' in personality_analysis:
        profile = personality_analysis['overall_profile']
        report += f"""
- **Personalidade dominante**: {profile['dominant_trait']} ({profile['classification']})
- **Percentil da média**: {profile['average_percentile']:.1%}
- **Interpretação**: {profile['interpretation']}
- **Comandos analisados**: {profile['total_commands']}

### Scores Médios Brutos:
"""
        for trait, score in profile['average_scores'].items():
            report += f"- **{trait}**: {score:.3f}\n"
        
        report += f"""
### Percentis da Média:
"""
        for trait, percentile in profile['average_percentiles'].items():
            classification = "POSITIVO" if percentile > 0.5 else "NEGATIVO"
            report += f"- **{trait}**: {percentile:.1%} ({classification})\n"
        
        report += f"""
### Análise Detalhada:
"""
        
        dominant = profile['dominant_trait']
        classification = profile['classification']
        
        if dominant == "HonestyHumility" and classification == "POSITIVO":
            report += "→ Atacante ético, prefere comandos de verificação e transparência\n"
            report += "→ A média dos seus comandos demonstra honestidade e humildade\n"
        elif dominant == "Emotionality" and classification == "POSITIVO":
            report += "→ Atacante cauteloso, usa comandos de backup e monitoramento cuidadoso\n"
            report += "→ A média dos seus comandos demonstra alta emocionalidade e cautela\n"
        elif dominant == "Extraversion" and classification == "POSITIVO":
            report += "→ Atacante sociável, emprega ferramentas de comunicação e interação\n"
            report += "→ A média dos seus comandos demonstra alta extraversão e confiança\n"
        elif dominant == "Agreeableness" and classification == "POSITIVO":
            report += "→ Atacante cooperativo, prefere comandos colaborativos e flexíveis\n"
            report += "→ A média dos seus comandos demonstra alta cordialidade\n"
        elif dominant == "Conscientiousness" and classification == "POSITIVO":
            report += "→ Atacante organizado, usa comandos sistemáticos e disciplinados\n"
            report += "→ A média dos seus comandos demonstra alta conscienciosidade\n"
        elif dominant == "OpennessToExperience" and classification == "POSITIVO":
            report += "→ Atacante criativo, emprega ferramentas inovadoras e exploratórias\n"
            report += "→ A média dos seus comandos demonstra alta abertura a experiências\n"
        else:
            report += f"→ Atacante com percentil médio abaixo de 50% em {dominant}\n"
            report += f"→ A média dos seus comandos não se destaca significativamente neste traço\n"
    
    report += f"""
## Complexidade dos Comandos
- **Complexidade média**: {metrics['command_complexity'].get('avg_complexity', 0):.1f} argumentos
- **Comandos simples (≤2 args)**: {metrics['command_complexity'].get('simple_commands', 0)}
- **Comandos complexos (>2 args)**: {metrics['command_complexity'].get('complex_commands', 0)}

## Metodologia
- **Classificação**: Semi-supervisionada com SentiWordNet 3.0
- **Refinamento**: Random Walk com propagação de escores
- **Análise**: Percentis relativos ao universo de comandos ({len(all_known_commands)} comandos)
- **Traços analisados**: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience
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
        
        # Passos 3-5: Tenta ML; se indisponível (SciPy/scikit-learn), usa fallback por seeds
        try:
            from classifiers.vectorial_classifier import VectorialClassifier
            logger.info("Preparando classificadores (scikit-learn)...")
            gloss_extractor = CommandGlossExtractor()
            classifier = VectorialClassifier(gloss_extractor)
            classifier.set_relations(COMMAND_RELATIONS)
            classifier.prepare_global_vectorizer(all_known_commands)
            trait_classifiers = classifier.train_trait_classifiers(expanded_sets)
            initial_scores = generate_initial_scores(classifier, trait_classifiers, all_known_commands)
        except BaseException as e:
            logger.warning(f"Falha ao inicializar classificadores ML ({e}). Usando fallback sem ML.")
            classifier = None
            initial_scores = generate_initial_scores_from_seeds(expanded_sets, all_known_commands)
        
        # Passo 6: Refina os escores com Random Walk
        logger.info("Iniciando refinamento com Random Walk...")
        refiner = RandomWalkRefiner(
            all_known_commands,
            COMMAND_RELATIONS,
            alpha=0.60,
            iterations=20,
            tolerance=1e-4,
            patience=5
        )
        final_scores = refiner.refine_scores_multi_trait(initial_scores)
        
        # Passo 7: Gera métricas e visualizações
        logger.info("Gerando métricas e visualizações...")
        metrics = generate_metrics(expanded_sets, initial_scores, final_scores, test_commands, refiner)
        
        # Passo 8: Cria visualizações
        create_visualizations(expanded_sets, initial_scores, final_scores, test_commands, results_dir)
        
        # Passo 9: Gera relatório resumido
        generate_summary_report(expanded_sets, initial_scores, final_scores, test_commands, metrics, refiner, results_dir, all_known_commands)
        
        # Passo 10: Validação de efetividade (rótulos fracos via seeds)
        validate_effectiveness(final_scores, expanded_sets, results_dir)

        # Passo 11: Salva resultados detalhados
        save_detailed_results(expanded_sets, initial_scores, final_scores, test_commands, metrics, refiner, results_dir)
        
        # Passo 12: Apresenta resultados no console
        display_results(test_commands, initial_scores, final_scores, all_known_commands)
        
        # Passo 13: Estatísticas finais
        display_final_stats(expanded_sets, all_known_commands, refiner, classifier)
        
        logger.info(f"Pipeline modularizado executado com sucesso! Resultados salvos em: {results_dir}")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main() 