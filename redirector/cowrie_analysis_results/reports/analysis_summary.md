# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

## Resumo Executivo
- **Total de comandos analisados**: 681
- **Comandos de teste**: 18
- **Iterações Random Walk**: 25
- **Fator de amortecimento (α)**: 0.7

## Expansão dos Conjuntos Semente
- **HonestyHumility_Positive**: 197 comandos
- **HonestyHumility_Negative**: 241 comandos
- **Emotionality_Positive**: 303 comandos
- **Emotionality_Negative**: 249 comandos
- **Extraversion_Positive**: 224 comandos
- **Extraversion_Negative**: 313 comandos
- **Agreeableness_Positive**: 311 comandos
- **Agreeableness_Negative**: 259 comandos
- **Conscientiousness_Positive**: 243 comandos
- **Conscientiousness_Negative**: 313 comandos
- **OpennessToExperience_Positive**: 259 comandos
- **OpennessToExperience_Negative**: 323 comandos

## Métricas de Classificação
- **Melhoria média nos escores**: 0.078
- **Classificações com melhoria positiva**: 71/108
- **Melhoria máxima**: 0.317

## Distribuição de Personalidade (Percentis)
- **HonestyHumility**: 0 comandos (0.0%)
- **Emotionality**: 10 comandos (55.6%)
- **Extraversion**: 0 comandos (0.0%)
- **Agreeableness**: 8 comandos (44.4%)
- **Conscientiousness**: 0 comandos (0.0%)
- **OpennessToExperience**: 0 comandos (0.0%)

## Perfil do Atacante (Baseado em Percentil da Média)

- **Personalidade dominante**: Extraversion (POSITIVO)
- **Percentil da média**: 91.6%
- **Interpretação**: A média do atacante é mais extraversion que 92% dos comandos
- **Comandos analisados**: 18

### Scores Médios Brutos:
- **HonestyHumility**: 0.294
- **Emotionality**: 0.497
- **Extraversion**: 0.282
- **Agreeableness**: 0.490
- **Conscientiousness**: 0.305
- **OpennessToExperience**: 0.391

### Percentis da Média:
- **HonestyHumility**: 79.3% (POSITIVO)
- **Emotionality**: 79.6% (POSITIVO)
- **Extraversion**: 91.6% (POSITIVO)
- **Agreeableness**: 81.6% (POSITIVO)
- **Conscientiousness**: 91.3% (POSITIVO)
- **OpennessToExperience**: 82.5% (POSITIVO)

### Análise Detalhada:
→ Atacante sociável, emprega ferramentas de comunicação e interação
→ A média dos seus comandos demonstra alta extraversão e confiança

## Complexidade dos Comandos
- **Complexidade média**: 0.9 argumentos
- **Comandos simples (≤2 args)**: 18
- **Comandos complexos (>2 args)**: 0

## Metodologia
- **Classificação**: Semi-supervisionada com SentiWordNet 3.0
- **Refinamento**: Random Walk com propagação de escores
- **Análise**: Percentis relativos ao universo de comandos (681 comandos)
- **Traços analisados**: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience
- **Vectorizer**: TF-IDF global com gloss dos comandos

---
*Relatório gerado automaticamente pelo Pipeline de Classificação Cowrie*
