# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

## Resumo Executivo
- **Total de comandos analisados**: 876
- **Comandos de teste**: 18
- **Iterações Random Walk**: 25
- **Fator de amortecimento (α)**: 0.65

## Expansão dos Conjuntos Semente
- **HonestyHumility_Positive**: 182 comandos
- **HonestyHumility_Negative**: 170 comandos
- **Emotionality_Positive**: 170 comandos
- **Emotionality_Negative**: 140 comandos
- **Extraversion_Positive**: 181 comandos
- **Extraversion_Negative**: 206 comandos
- **Agreeableness_Positive**: 161 comandos
- **Agreeableness_Negative**: 164 comandos
- **Conscientiousness_Positive**: 191 comandos
- **Conscientiousness_Negative**: 172 comandos
- **OpennessToExperience_Positive**: 196 comandos
- **OpennessToExperience_Negative**: 187 comandos

## Métricas de Classificação
- **Melhoria média nos escores**: 0.253
- **Classificações com melhoria positiva**: 79/108
- **Melhoria máxima**: 0.862

## Distribuição de Personalidade (Percentis)
- **HonestyHumility**: 3 comandos (16.7%)
- **Emotionality**: 6 comandos (33.3%)
- **Extraversion**: 1 comandos (5.6%)
- **Agreeableness**: 1 comandos (5.6%)
- **Conscientiousness**: 1 comandos (5.6%)
- **OpennessToExperience**: 6 comandos (33.3%)

## Perfil do Atacante (Baseado em Percentil da Média)

- **Personalidade dominante**: Agreeableness (POSITIVO)
- **Percentil da média**: 68.8%
- **Interpretação**: A média do atacante é mais agreeableness que 69% dos comandos
- **Comandos analisados**: 18

### Scores Médios Brutos:
- **HonestyHumility**: 0.387
- **Emotionality**: 0.579
- **Extraversion**: 0.429
- **Agreeableness**: 0.438
- **Conscientiousness**: 0.282
- **OpennessToExperience**: 0.545

### Percentis da Média:
- **HonestyHumility**: 46.7% (NEGATIVO)
- **Emotionality**: 63.5% (POSITIVO)
- **Extraversion**: 61.0% (POSITIVO)
- **Agreeableness**: 68.8% (POSITIVO)
- **Conscientiousness**: 43.6% (NEGATIVO)
- **OpennessToExperience**: 58.8% (POSITIVO)

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
- **Análise**: Percentis relativos ao universo de comandos (876 comandos)
- **Traços analisados**: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience
- **Vectorizer**: TF-IDF global com gloss dos comandos
 - **Validação**: CV estratificada com previsões OOF e ICs por bootstrap; gráficos ROC/PR, calibração (Brier/ECE), matriz de confusão e varredura de limiar por traço

---
*Relatório gerado automaticamente pelo Pipeline de Classificação Cowrie*
