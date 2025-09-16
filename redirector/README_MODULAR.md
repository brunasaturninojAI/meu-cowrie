# Classificação Semi-Supervisionada de Comandos - Versão Modularizada

## 📁 Estrutura do Projeto

```
redirector/
├── config/                          # Configurações e dados
│   ├── __init__.py
│   ├── seed_sets.py                 # Conjuntos semente para cada traço
│   └── command_relations.py         # Relações taxonômicas entre comandos
├── extractors/                      # Extratores de dados
│   ├── __init__.py
│   └── gloss_extractor.py          # Extrai descrições das páginas man
├── expanders/                       # Expansores de conjuntos
│   ├── __init__.py
│   └── semi_supervised_expander.py # Expansão semi-supervisionada
├── classifiers/                     # Classificadores
│   ├── __init__.py
│   └── vectorial_classifier.py     # Classificador TF-IDF + SVM
├── refiners/                        # Refinadores de escores
│   ├── __init__.py
│   └── random_walk_refiner.py      # Refinamento com Random Walk
├── main.py                          # Pipeline principal
├── semisupervised_expansion.py      # Versão original (monolítica)
└── README_MODULAR.md               # Este arquivo
```

## 🚀 Como Executar

### Versão Modularizada (Recomendada)
```bash
cd redirector
source venv/bin/activate
python main.py
```

### Versão Original (Monolítica)
```bash
cd redirector
source venv/bin/activate
python semisupervised_expansion.py
```

## 🔧 Módulos e Responsabilidades

### 1. **Config** (`config/`)
- **`seed_sets.py`**: Conjuntos semente para cada traço psicológico
- **`command_relations.py`**: Relações taxonômicas (similar, antonym, derived_from, also_see)

### 2. **Extractors** (`extractors/`)
- **`gloss_extractor.py`**: Extrai descrições das páginas man (glossas)

### 3. **Expanders** (`expanders/`)
- **`semi_supervised_expander.py`**: Expande conjuntos semente usando relações taxonômicas

### 4. **Classifiers** (`classifiers/`)
- **`vectorial_classifier.py`**: Classificadores TF-IDF + SVM para cada traço

### 5. **Refiners** (`refiners/`)
- **`random_walk_refiner.py`**: Refina escores usando algoritmo de Random Walk

### 6. **Main** (`main.py`)
- Orquestra todo o pipeline de classificação

## 📊 Vantagens da Modularização

### ✅ **Manutenibilidade**
- Cada módulo tem responsabilidade única
- Fácil localizar e corrigir problemas
- Código mais legível e organizado

### ✅ **Reutilização**
- Módulos podem ser importados independentemente
- Fácil criar novos pipelines ou experimentos
- Testes unitários mais simples

### ✅ **Extensibilidade**
- Adicionar novos classificadores é simples
- Novos métodos de refinamento podem ser implementados
- Configurações podem ser facilmente modificadas

### ✅ **Colaboração**
- Diferentes desenvolvedores podem trabalhar em módulos diferentes
- Menos conflitos de merge
- Código mais profissional

## 🔄 Migração da Versão Original

A versão modularizada mantém **100% de compatibilidade** com a versão original:

- **Mesmos resultados**: Output idêntico
- **Mesmos parâmetros**: Configurações preservadas
- **Mesma performance**: Sem overhead adicional

## 🧪 Testando Módulos Individualmente

```python
# Testar apenas o extrator de glossas
from extractors.gloss_extractor import CommandGlossExtractor
extractor = CommandGlossExtractor()
gloss = extractor.get_command_gloss("ls")
print(gloss)

# Testar apenas o expansor
from config.seed_sets import SEED_SETS
from config.command_relations import COMMAND_RELATIONS
from expanders.semi_supervised_expander import SemiSupervisedExpander

expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS)
expanded = expander.expand_seeds()
print(f"Comandos expandidos: {len(expanded)}")
```

## 📈 Próximos Passos

1. **Testes unitários** para cada módulo
2. **Documentação de API** para cada classe
3. **Configuração via arquivo** (YAML/JSON)
4. **Interface de linha de comando** (CLI)
5. **Interface web** para visualização de resultados

## 🤝 Contribuindo

Para contribuir com melhorias:

1. **Mantenha a modularização**: Cada funcionalidade em seu módulo
2. **Documente mudanças**: Atualize docstrings e comentários
3. **Teste localmente**: Execute `python main.py` antes de commitar
4. **Mantenha compatibilidade**: Não quebre a API existente

## 📝 Exemplo de Uso Avançado

```python
from config.seed_sets import SEED_SETS
from config.command_relations import COMMAND_RELATIONS
from extractors.gloss_extractor import CommandGlossExtractor
from expanders.semi_supervised_expander import SemiSupervisedExpander
from classifiers.vectorial_classifier import VectorialClassifier
from refiners.random_walk_refiner import RandomWalkRefiner

# Pipeline customizado
extractor = CommandGlossExtractor()
expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS, max_iterations=5)
classifier = VectorialClassifier(extractor)

# Expande conjuntos
expanded_sets = expander.expand_seeds()

# Treina classificadores
classifier.prepare_global_vectorizer(list(expanded_sets.values())[0])
trait_classifiers = classifier.train_trait_classifiers(expanded_sets)

# Classifica comando específico
result = classifier.classify_command("grep -r 'pattern' .", trait_classifiers)
print(result)
```

---

**Versão**: 2.0 (Modularizada)  
**Data**: 2024  
**Metodologia**: SentiWordNet 3.0 + Random Walk  
**Arquitetura**: Módulos Python com responsabilidades bem definidas 