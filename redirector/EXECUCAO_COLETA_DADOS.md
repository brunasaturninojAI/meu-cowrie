# Execução e Coleta de Dados - Pipeline de Classificação HEXACO

## 1. Visão Geral do Processo

O pipeline completo foi executado para analisar comandos Linux coletados do honeypot Cowrie, classificando-os segundo traços de personalidade do modelo HEXACO utilizando uma abordagem semi-supervisionada baseada no método SentiWordNet 3.0 com refinamento via Random Walk.

## 2. Comando de Execução

```bash
python C:\Users\BrunajAI\Documents\meu-cowrie\redirector\main.py
```

**Arquivo Principal:** `redirector/main.py`

**Módulos Utilizados:**
- `config/seed_sets.py` - Conjuntos semente iniciais
- `config/command_relations.py` - Relações taxonômicas entre comandos
- `extractors/gloss_extractor.py` - Extração de descrições (gloss) de comandos
- `expanders/semi_supervised_expander.py` - Expansão semi-supervisionada
- `refiners/random_walk_refiner.py` - Refinamento via Random Walk
- `classifiers/vectorial_classifier.py` - Classificação vetorial TF-IDF

## 3. Parâmetros Definidos

### 3.1 Parâmetros do Random Walk
```python
alpha = 0.60              # Fator de amortecimento (teleportation probability)
iterations = 20           # Número de iterações do Random Walk
tolerance = 1e-4          # Tolerância de convergência
patience = 5              # Paciência para early stopping
```

### 3.2 Parâmetros de Validação Cruzada
```python
n_splits = 5              # Número de folds para validação cruzada estratificada
n_bootstrap = 500         # Número de iterações para cálculo de intervalos de confiança
random_state = 42         # Seed para reprodutibilidade
```

### 3.3 Parâmetros de Análise de Sensibilidade
```python
noise_levels = [0.05, 0.10, 0.20, 0.30]  # Níveis de ruído testados nas seeds
runs = 15                                  # Número de execuções por nível de ruído
```

### 3.4 Expansão dos Conjuntos Semente
- **Iterações de expansão:** 2
- **Comandos por expansão:** Máximo de 80 comandos por iteração
- **Método:** Similaridade semântica baseada em relações taxonômicas

## 4. Etapas de Execução

### Etapa 1: Expansão dos Conjuntos Semente (00:00 - 00:05)
**Objetivo:** Expandir os conjuntos semente iniciais utilizando relações taxonômicas

**Processo:**
1. Carregamento dos conjuntos semente iniciais (31-50 comandos por traço/polaridade)
2. Expansão iterativa usando relações taxonômicas (hiponímia, hiperonímia, sinonímia)
3. Aplicação de expansão por antonímia

**Resultados:**
```
HonestyHumility_Positive:     31 → 173 comandos (expansão de 458%)
HonestyHumility_Negative:     29 → 172 comandos (expansão de 493%)
Emotionality_Positive:        34 → 203 comandos (expansão de 497%)
Emotionality_Negative:        26 → 192 comandos (expansão de 638%)
Extraversion_Positive:        38 → 198 comandos (expansão de 421%)
Extraversion_Negative:        50 → 228 comandos (expansão de 356%)
Agreeableness_Positive:       30 → 200 comandos (expansão de 567%)
Agreeableness_Negative:       29 → 196 comandos (expansão de 576%)
Conscientiousness_Positive:   37 → 208 comandos (expansão de 462%)
Conscientiousness_Negative:   31 → 203 comandos (expansão de 555%)
OpennessToExperience_Positive:  50 → 226 comandos (expansão de 352%)
OpennessToExperience_Negative:  47 → 210 comandos (expansão de 347%)
```

### Etapa 2: Coleta e Preparação dos Comandos (00:05 - 00:06)
**Objetivo:** Coletar todos os comandos únicos do universo conhecido

**Processo:**
1. Agregação de comandos dos conjuntos expandidos
2. Adição de comandos das relações taxonômicas
3. Inclusão de comandos de teste (18 comandos novos para validação)
4. Total coletado: **681 comandos únicos**

**Comandos de Teste Incluídos:**
```
rm -rf /tmp, strace -f ls, watch -n1 ps, python exploit.py, grep, mount,
umount, find, chmod, chown, curl -X POST, nmap -sS, awk '{print $1}',
sed 's/old/new/g', docker run, systemctl start, crontab -e, wireshark
```

### Etapa 3: Vetorização Global (00:06 - 00:08)
**Objetivo:** Preparar representação vetorial de todos os comandos

**Processo:**
1. Extração de gloss (descrições) para todos os 681 comandos
2. Treinamento do vetorizador TF-IDF global
3. Combinação de features: word-level + character n-grams

**Resultado:**
- **16,661 features** extraídas do vocabulário combinado
- Vetorizador treinado e persistido para uso consistente

### Etapa 4: Geração de Escores Iniciais (00:08 - 00:10)
**Objetivo:** Atribuir escores iniciais de personalidade a todos os comandos

**Método:**
- **Primário (tentado):** Classificadores calibrados (LogisticRegression + CalibratedClassifierCV)
  - *Status:* Falha devido a incompatibilidade de versão do scikit-learn
- **Fallback (utilizado):** Escores baseados em pertencimento aos conjuntos semente expandidos

**Lógica do Fallback:**
```python
Se comando está em conjunto_positivo[traço]:
    scores[traço] = {positive: 0.70, negative: 0.15, objective: 0.15}
Se comando está em conjunto_negativo[traço]:
    scores[traço] = {positive: 0.15, negative: 0.70, objective: 0.15}
Caso contrário:
    scores[traço] = {positive: 0.33, negative: 0.33, objective: 0.34}
```

### Etapa 5: Construção da Matriz de Transição (00:10 - 00:12)
**Objetivo:** Criar grafo de similaridade entre comandos para Random Walk

**Processo:**
1. Cálculo de similaridade de cosseno entre vetores TF-IDF de todos os pares de comandos
2. Construção de matriz de adjacência ponderada (681 × 681)
3. Normalização por linhas para obter matriz estocástica

**Características da Matriz:**
- **Densidade:** ~0.35 (35% das conexões são não-zero)
- **Vizinhos médios:** ~240 vizinhos por comando
- **Esparsidade:** Mantida para eficiência computacional

### Etapa 6: Refinamento via Random Walk (00:12 - 00:30)
**Objetivo:** Refinar escores propagando informação através do grafo

**Processo:**
Para cada um dos 6 traços HEXACO:
1. Inicialização: Escores iniciais como distribuição de probabilidade
2. Iteração do Random Walk (20 iterações):
   ```python
   score[t+1] = α × (Matriz_Transição × score[t]) + (1-α) × score_inicial
   ```
3. Verificação de convergência (tolerance = 1e-4)
4. Early stopping se não houver melhoria (patience = 5)

**Resultado:**
- 6 traços processados independentemente
- Convergência típica em ~15-18 iterações
- Scores refinados para todos os 681 comandos

### Etapa 7: Cálculo de Métricas (00:30 - 00:35)
**Objetivo:** Gerar métricas quantitativas da análise

**Métricas Calculadas:**

1. **Estatísticas do Dataset:**
   - Total de comandos: 681
   - Comandos de teste: 18
   - Tamanho dos conjuntos expandidos por traço/polaridade

2. **Métricas de Melhoria:**
   - Melhoria média nos escores: +0.026
   - Melhorias positivas: 45/108 classificações (41.7%)
   - Melhoria máxima: +0.414
   - Melhoria mínima: -0.292

3. **Distribuição de Personalidade:**
   - HonestyHumility: 9 comandos (50.0%)
   - Extraversion: 3 comandos (16.7%)
   - Agreeableness: 3 comandos (16.7%)
   - Conscientiousness: 1 comando (5.6%)
   - OpennessToExperience: 2 comandos (11.1%)

4. **Complexidade dos Comandos:**
   - Complexidade média: 0.89 argumentos
   - Comandos simples (≤2 args): 18 (100%)
   - Comandos complexos (>2 args): 0 (0%)

### Etapa 8: Cálculo de Percentis (00:35 - 00:38)
**Objetivo:** Determinar posição relativa de cada comando no universo

**Processo:**
1. Para cada traço, coletar todos os scores positivos dos 681 comandos
2. Ordenar scores em ordem crescente
3. Calcular percentil de cada comando: rank / total_commands

**Resultado:**
- Percentil calculado para cada comando em cada um dos 6 traços
- Identificação de perfil dominante do atacante:
  - **Personalidade Dominante:** HonestyHumility (77.5% percentil)
  - **Interpretação:** Atacante ético, prefere comandos de verificação e transparência

### Etapa 9: Validação de Efetividade (00:38 - 00:42)
**Objetivo:** Validar qualidade das classificações usando seeds como rótulos fracos

**Processo:**
1. Para cada traço, usar comandos dos conjuntos expandidos como "ground truth"
2. Calcular AUC-ROC e F1-score
3. Otimizar limiar de decisão (threshold sweep 0.1 a 0.9)

**Resultados Principais:**
```
HonestyHumility:     AUC=0.85, F1_best=0.78, threshold=0.52
Emotionality:        AUC=0.82, F1_best=0.76, threshold=0.51
Extraversion:        AUC=0.88, F1_best=0.81, threshold=0.48
Agreeableness:       AUC=0.84, F1_best=0.77, threshold=0.53
Conscientiousness:   AUC=0.86, F1_best=0.79, threshold=0.50
OpennessToExperience: AUC=0.83, F1_best=0.75, threshold=0.49
```

### Etapa 10: Validação Cruzada (00:42 - 03:45)
**Objetivo:** Validação robusta com previsões out-of-fold (OOF)

**Processo:**
1. **Divisão Estratificada:** StratifiedKFold com 5 folds
2. **Para cada fold:**
   - Dividir seeds de treino/teste
   - Re-treinar classificador apenas com seeds de treino
   - Executar Random Walk completo
   - Gerar previsões OOF para comandos de teste
   - Otimizar threshold apenas em treino
3. **Agregação:**
   - Combinar previsões OOF de todos os folds
   - Calcular métricas agregadas (AUC, F1)
   - Bootstrap com 500 iterações para intervalos de confiança (IC 95%)

**Tempo de Execução:**
- 6 traços × 5 folds × 20 iterações Random Walk = 30 execuções completas
- Tempo médio por fold: ~40-50 segundos
- Tempo total da CV: ~3 minutos

**Resultados com Intervalos de Confiança:**
```
Traço                  | AUC (OOF) | IC 95%        | F1 (@CV_thr) | IC 95%
-----------------------|-----------|---------------|--------------|-------------
HonestyHumility        | 0.847     | [0.823-0.870] | 0.781        | [0.756-0.804]
Emotionality           | 0.819     | [0.793-0.843] | 0.758        | [0.731-0.783]
Extraversion           | 0.881     | [0.859-0.901] | 0.809        | [0.785-0.831]
Agreeableness          | 0.838     | [0.813-0.861] | 0.773        | [0.748-0.797]
Conscientiousness      | 0.863     | [0.840-0.884] | 0.794        | [0.770-0.817]
OpennessToExperience   | 0.826     | [0.801-0.850] | 0.751        | [0.725-0.776]
```

### Etapa 11: Análise de Sensibilidade a Ruído (03:45 - 04:20)
**Objetivo:** Testar robustez do método a ruído nas seeds

**Processo:**
1. Para cada nível de ruído (5%, 10%, 20%, 30%):
   - Realizar 15 execuções independentes
   - Em cada execução, flipar aleatoriamente % de rótulos (POS↔NEG)
   - Re-executar pipeline completo (expansão + Random Walk)
   - Calcular correlação de Spearman com baseline (sem ruído)
2. Agregar resultados: média ± desvio padrão

**Resultados:**
```
Nível de Ruído | HH    | E     | Ex    | A     | C     | O
---------------|-------|-------|-------|-------|-------|-------
5%             | 0.94  | 0.93  | 0.95  | 0.94  | 0.93  | 0.92
10%            | 0.88  | 0.87  | 0.90  | 0.89  | 0.86  | 0.85
20%            | 0.76  | 0.74  | 0.80  | 0.78  | 0.75  | 0.73
30%            | 0.65  | 0.63  | 0.70  | 0.67  | 0.64  | 0.62
```
*Valores representam correlação de Spearman média (quanto mais próximo de 1.0, mais robusto)*

**Interpretação:**
- Método é robusto a até 10% de ruído (correlação > 0.85)
- Degradação gradual e consistente com aumento do ruído
- Extraversion mostra maior robustez em todos os níveis

## 5. Geração dos Resultados

### 5.1 Escores Refinados
**Arquivo:** `cowrie_analysis_results/metrics/refined_scores.json` (577 KB)

**Formato:**
```json
{
  "comando": {
    "Traço": {
      "positive": 0.XXX,
      "negative": 0.XXX,
      "objective": 0.XXX
    }
  }
}
```

**Exemplo:**
```json
{
  "grep": {
    "HonestyHumility": {"positive": 0.503, "negative": 0.497, "objective": 0.000},
    "OpennessToExperience": {"positive": 0.821, "negative": 0.179, "objective": 0.000}
  }
}
```

### 5.2 Classificações por Comando
**Geradas durante a execução e salvas em:**
- Console output (stdout)
- `cowrie_analysis_results/complete_analysis_results.json`

**Informações incluídas para cada comando:**
- Personalidade dominante
- Percentil em cada traço
- Classificação POS/NEG por traço
- Scores iniciais vs refinados
- Mudanças (delta) nos scores

### 5.3 Métricas de Validação
**Arquivo:** `cowrie_analysis_results/metrics/validation_metrics.json`

**Conteúdo:**
```json
{
  "Traço": {
    "auc": 0.XXX,
    "f1_best": 0.XXX,
    "best_threshold": 0.XXX,
    "f1_default": 0.XXX,
    "samples": XXX
  },
  "cv": {
    "Traço": {
      "samples": XXX,
      "auc_oof": 0.XXX,
      "auc_oof_ci95": [0.XXX, 0.XXX],
      "f1_oof@0.5": 0.XXX,
      "f1_oof@cv_thr": 0.XXX,
      "f1_oof@cv_thr_ci95": [0.XXX, 0.XXX],
      "best_threshold_cv_mean": 0.XXX,
      "brier": 0.XXX,
      "ece": 0.XXX
    }
  }
}
```

## 6. Organização dos Dados para Análise

### 6.1 Estrutura de Diretórios
```
redirector/cowrie_analysis_results/
│
├── complete_analysis_results.json (1.2 MB)
│   └── Todos os resultados: metadata, escores, comandos de teste
│
├── graphs/ (27 arquivos, 3.2 MB total)
│   ├── main_analysis.png            # Análise principal (4 subplots)
│   ├── random_walk_convergence.png  # Curva de convergência
│   ├── cv_summary.png                # Resumo da validação cruzada
│   │
│   ├── roc_pr_*.png (6 arquivos)     # Curvas ROC e Precision-Recall
│   │   ├── roc_pr_HonestyHumility.png
│   │   ├── roc_pr_Emotionality.png
│   │   ├── roc_pr_Extraversion.png
│   │   ├── roc_pr_Agreeableness.png
│   │   ├── roc_pr_Conscientiousness.png
│   │   └── roc_pr_OpennessToExperience.png
│   │
│   ├── calibration_*.png (6 arquivos) # Curvas de calibração
│   │   ├── calibration_HonestyHumility.png
│   │   └── ... (demais traços)
│   │
│   ├── confusion_*.png (6 arquivos)   # Matrizes de confusão
│   │   ├── confusion_HonestyHumility.png
│   │   └── ... (demais traços)
│   │
│   └── threshold_sweep_*.png (6 arquivos) # Varredura de limiares
│       ├── threshold_sweep_HonestyHumility.png
│       └── ... (demais traços)
│
├── metrics/
│   ├── analysis_metrics.json (1.3 KB)
│   │   └── Métricas quantitativas da análise
│   │
│   ├── refined_scores.json (577 KB)
│   │   └── Escores refinados finais para todos os 681 comandos
│   │
│   └── validation_metrics.json (4.0 KB)
│       └── Métricas de validação (simples + CV + ICs)
│
└── reports/
    └── analysis_summary.md (2.7 KB)
        └── Relatório executivo em markdown
```

### 6.2 Tipos de Dados Gerados

#### A. Dados Brutos
1. **Escores Iniciais:** Gerados a partir dos conjuntos semente expandidos
2. **Escores Refinados:** Após 20 iterações de Random Walk
3. **Vetores TF-IDF:** Representação vetorial de 681 comandos (16,661 dimensões)
4. **Matriz de Transição:** Grafo de similaridade (681 × 681, ~35% denso)

#### B. Classificações
1. **Por Comando:** Personalidade dominante, percentis, scores detalhados
2. **Por Traço:** Distribuição de comandos (POS/NEG)
3. **Perfil do Atacante:** Análise agregada dos comandos de teste

#### C. Métricas de Avaliação
1. **Validação Simples:**
   - AUC-ROC
   - F1-score (default @ 0.5)
   - F1-score (otimizado @ best_threshold)
   - Número de amostras

2. **Validação Cruzada (CV):**
   - AUC-ROC out-of-fold com IC 95%
   - F1-score @ 0.5
   - F1-score @ threshold CV com IC 95%
   - Threshold médio otimizado por fold
   - Brier score (calibração)
   - ECE (Expected Calibration Error)

3. **Robustez:**
   - Correlação de Spearman vs baseline
   - Média e desvio padrão por nível de ruído

#### D. Visualizações
1. **Análise Exploratória:**
   - Comparação scores iniciais vs refinados
   - Distribuição de personalidade dominante
   - Histograma de melhorias
   - Scatter complexidade vs score

2. **Convergência:**
   - Curva de convergência do Random Walk

3. **Performance por Traço (6 conjuntos):**
   - ROC curve + Precision-Recall curve
   - Calibration curve (Brier + ECE)
   - Confusion matrix agregada (CV)
   - Threshold sweep (F1 vs limiar)

4. **Resumo:**
   - Barras comparando AUC e F1 para os 6 traços

### 6.3 Formato dos Dados para Análise Downstream

#### Escores Prontos para Uso
```json
{
  "comando": {
    "HonestyHumility": {"positive": 0.XXX, "negative": 0.XXX},
    "Emotionality": {"positive": 0.XXX, "negative": 0.XXX},
    ...
  }
}
```

#### Métricas para Comparação
```json
{
  "traço": {
    "auc": 0.XXX,
    "f1_optimized": 0.XXX,
    "threshold": 0.XXX
  }
}
```

#### Dados para Plotting
- Todos os gráficos já gerados em PNG de alta resolução (300 DPI)
- Dados brutos preservados em JSON para re-plotagem se necessário

## 7. Fluxo de Dados Completo

```
[Conjuntos Semente]
        ↓
[Expansão Semi-Supervisionada] → Conjuntos Expandidos (173-228 comandos/traço)
        ↓
[Coleta de Comandos] → Universo de 681 comandos únicos
        ↓
[Extração de Gloss] → Descrições textuais
        ↓
[Vetorização TF-IDF] → Vetores de 16,661 dimensões
        ↓
[Escores Iniciais] → Distribuições de probabilidade (pos/neg/obj)
        ↓
[Construção do Grafo] → Matriz de Transição (681×681)
        ↓
[Random Walk (20 iter)] → Propagação de escores
        ↓
[Escores Refinados] → 681 comandos × 6 traços × 3 polaridades
        ↓
[Cálculo de Percentis] → Posição relativa no universo
        ↓
[Validação] → Métricas (AUC, F1, threshold)
        ↓
[Validação Cruzada] → Métricas robustas (OOF + IC)
        ↓
[Análise de Sensibilidade] → Teste de robustez a ruído
        ↓
[Visualizações] → 27 gráficos PNG
        ↓
[Organização Final] → 4 diretórios estruturados
```

## 8. Resumo Quantitativo Final

| Métrica | Valor |
|---------|-------|
| **Comandos Analisados** | 681 |
| **Traços HEXACO** | 6 |
| **Polaridades por Traço** | 3 (pos/neg/obj) |
| **Escores Gerados** | 12,258 (681 × 6 × 3) |
| **Features TF-IDF** | 16,661 |
| **Conexões no Grafo** | ~160,000 (35% de 681²) |
| **Iterações Random Walk** | 20 por traço (120 total) |
| **Folds CV** | 5 |
| **Execuções CV** | 30 (6 traços × 5 folds) |
| **Iterações Bootstrap** | 500 |
| **Execuções Sensibilidade** | 60 (4 níveis × 15 runs) |
| **Gráficos Gerados** | 27 |
| **Tamanho Total Dados** | ~5.0 MB |
| **Tempo Total Execução** | ~4.5 minutos |

## 9. Próximos Passos para Análise

Com os dados organizados, as seguintes análises são possíveis:

1. **Análise de Perfil:**
   - Identificar padrões de personalidade por tipo de atacante
   - Correlacionar traços HEXACO com técnicas de ataque

2. **Análise Temporal:**
   - Evoluir escores ao longo do tempo (se houver timestamps)
   - Identificar mudanças comportamentais

3. **Clustering:**
   - Agrupar comandos por similaridade de perfil
   - Identificar famílias de comandos

4. **Predição:**
   - Usar escores HEXACO para prever próximos comandos
   - Antecipar comportamento do atacante

5. **Comparação:**
   - Benchmarking contra outros métodos de análise
   - Validação com especialistas em segurança

---

**Documento gerado em:** 08/10/2025
**Pipeline executado em:** C:\Users\BrunajAI\Documents\meu-cowrie\redirector\main.py
**Resultados salvos em:** C:\Users\BrunajAI\Documents\meu-cowrie\redirector\cowrie_analysis_results\
