# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

## Resumo Executivo
- **Total de comandos analisados**: 695
- **Comandos de teste**: 18
- **Iterações Random Walk**: 25
- **Fator de amortecimento (α)**: 0.75

## Expansão dos Conjuntos Semente
- **HonestyHumility_Positive**: 83 comandos
- **HonestyHumility_Negative**: 69 comandos
- **Emotionality_Positive**: 72 comandos
- **Emotionality_Negative**: 70 comandos
- **Extraversion_Positive**: 79 comandos
- **Extraversion_Negative**: 105 comandos
- **Agreeableness_Positive**: 70 comandos
- **Agreeableness_Negative**: 70 comandos
- **Conscientiousness_Positive**: 92 comandos
- **Conscientiousness_Negative**: 83 comandos
- **OpennessToExperience_Positive**: 92 comandos
- **OpennessToExperience_Negative**: 90 comandos

## Métricas de Classificação
- **Melhoria média nos escores**: 0.086
- **Classificações com melhoria positiva**: 58/108
- **Melhoria máxima**: 0.434

## Distribuição de Personalidade (Percentis)
- **HonestyHumility**: 4 comandos (22.2%)
- **Emotionality**: 2 comandos (11.1%)
- **Extraversion**: 0 comandos (0.0%)
- **Agreeableness**: 2 comandos (11.1%)
- **Conscientiousness**: 2 comandos (11.1%)
- **OpennessToExperience**: 8 comandos (44.4%)

## Perfil do Atacante (Baseado em Percentil da Média)

- **Personalidade dominante**: Agreeableness (POSITIVO)
- **Percentil da média**: 83.0%
- **Interpretação**: A média do atacante é mais agreeableness que 83% dos comandos
- **Comandos analisados**: 18

### Scores Médios Brutos:
- **HonestyHumility**: 0.247
- **Emotionality**: 0.288
- **Extraversion**: 0.167
- **Agreeableness**: 0.288
- **Conscientiousness**: 0.224
- **OpennessToExperience**: 0.361

### Percentis da Média:
- **HonestyHumility**: 73.4% (POSITIVO)
- **Emotionality**: 78.6% (POSITIVO)
- **Extraversion**: 69.4% (POSITIVO)
- **Agreeableness**: 83.0% (POSITIVO)
- **Conscientiousness**: 71.5% (POSITIVO)
- **OpennessToExperience**: 81.7% (POSITIVO)

### Análise Detalhada:
→ Atacante cooperativo, prefere comandos colaborativos e flexíveis
→ A média dos seus comandos demonstra alta cordialidade

## Complexidade dos Comandos
- **Complexidade média**: 0.9 argumentos
- **Comandos simples (≤2 args)**: 18
- **Comandos complexos (>2 args)**: 0

## Metodologia
- **Classificação**: Semi-supervisionada com SentiWordNet 3.0
- **Refinamento**: Random Walk com propagação de escores
- **Análise**: Percentis relativos ao universo de comandos (695 comandos)
- **Traços analisados**: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience
- **Vectorizer**: TF-IDF global com gloss dos comandos
 - **Validação**: CV estratificada com previsões OOF e ICs por bootstrap; gráficos ROC/PR, calibração (Brier/ECE), matriz de confusão e varredura de limiar por traço

---
*Relatório gerado automaticamente pelo Pipeline de Classificação Cowrie*
