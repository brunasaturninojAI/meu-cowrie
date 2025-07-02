#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
# Script:       tree.py
# Finalidade:
#   - Sistema completo para criar perfis de atacantes e redirecioná-los.
#   - Utiliza uma taxonomia de comandos e a métrica de Wu & Palmer (1994)
#     para classificar sessões em perfis baseados em traços de personalidade.
#   - Redireciona para portas diferentes com base no perfil classificado.
#
# Fontes da Lógica:
#   - Fórmula de Similaridade: Wu & Palmer (1994), "VERB SEMANTICS AND LEXICAL SELECTION".
#   - Mapeamento de Comandos/Personalidades: "CPCS - comandos e personalidade.pdf".
#
# Uso:          sudo python3 comprehensive_persona_profiler.py
# -----------------------------------------------------------------

import json
import subprocess
import os

# --- PAINEL DE CONTROLE E CONFIGURAÇÃO ---

LOGFILE = "/home/brunasaturnino/meu-cowrie/cowrie1/var/log/cowrie/cowrie.json"
STATE_FILE = "comprehensive_profiler.state"
PORT_COWRIE1 = 2222


# 1. AÇÕES: Mapeamento dos Perfis Finais para as Portas de Destino
PROFILE_ACTION_PORTS = {
    "Descuidado_e_Impulsivo": 2224,
    "Metodico_e_Perfeccionista": 2226,
    "Explorador_Criativo": 2228,
    "Indefinido": None
}

# 2. PROTÓTIPOS: Nós da taxonomia que melhor representam cada traço de personalidade
PERSONA_PROTOTYPES = {
    "Perfeccionismo": "fsck",
    "Paciência": "watch",
    "Flexibilidade": "man",
    "Abertura_a_Experiencia": "gcc"
}

# 3. CONHECIMENTO: A Taxonomia Hierárquica de Comandos
COMMAND_TAXONOMY = {
    "Tatica_de_Comando": { # Raiz
        "Foco_em_Precisao": { # Traço: Perfeccionismo
            "Alta_Precisao": {
                "Verificacao_Sistema": {"dmesg": {}, "fsck": {}, "rpm -V": {}},
                "Analise_Detalhada": {"top": {}, "strace": {}, "find": {}, "grep": {}}
            },
            "Baixa_Precisao": {
                "Acao_Forcada": {"rm -rf": {}, "kill -9": {}},
                "Ignorar_Procedimentos": {"rpm --nodeeps": {}, "iptables -F": {}}
            }
        },
        "Foco_em_Tempo": { # Traço: Paciência
            "Acao_Metodica": {
                "Monitoramento_Continuo": {"watch": {}, "tcpdump": {}},
                "Busca_de_Conhecimento": {"man": {}, "apropos": {}}
            },
            "Acao_Imediata": {
                "Encerramento_Abrupto": {"kill -9": {}, "pkill": {}},
                "Remocao_Rapida": {"rm -rf": {}}
            }
        },
        "Foco_em_Adaptacao": { # Traço: Flexibilidade
            "Adaptacao_ao_Sistema": {
                "Coleta_de_Informacao": {"man": {}, "ps": {}, "netstat": {}},
                "Diagnostico_de_Rede": {"ping": {}, "dig": {}}
            },
            "Imposicao_no_Sistema": {
                "Acao_Inflexivel": {"rm -rf": {}, "kill -9": {}, "shutdown -h now": {}}
            }
        },
        "Foco_em_Exploracao": { # Traço: Abertura à Experiência
            "Criacao_e_Descoberta": {
                "Desenvolvimento": {"gcc": {}, "python": {}, "git": {}},
                "Exploracao_de_Rede": {"curl": {}, "wget": {}, "nc": {}, "ssh": {}}
            }
        }
    }
}

# --- MOTOR DE CÁLCULO: WU & PALMER (COM FUNÇÃO CORRIGIDA) ---

def find_path_to_root(taxonomy_dict, node_name):
    """
    Função para encontrar o caminho de um nó até a raiz (recursiva).
    Esta é a versão corrigida que navega corretamente na árvore.
    """
    # Itera sobre os nós do dicionário atual
    for parent, children in taxonomy_dict.items():
        # Se o nó que procuramos é um filho direto do nó atual...
        if node_name in children:
            # Encontrou! Retorna o caminho [nó, pai]
            return [node_name, parent]
        
        # Se não é um filho direto, mas o nó atual tem sub-dicionários...
        if isinstance(children, dict):
            # ...chama a função recursivamente para procurar nos sub-níveis.
            path = find_path_to_root(children, node_name)
            # Se a busca recursiva encontrou um caminho...
            if path:
                # ...adiciona o pai atual ao caminho e o retorna.
                path.append(parent)
                return path
    # Se não encontrou em nenhum lugar, retorna None.
    return None

def calculate_wu_palmer_similarity(taxonomy, node1, node2):
    """Implementação da fórmula de Wu & Palmer (1994)"""
    if node1 == node2: return 1.0
    
    # Adiciona a raiz ao final do caminho para garantir que sempre haja um caminho.
    path1 = find_path_to_root(taxonomy, node1)
    if path1: path1.append("Tatica_de_Comando")
    else: return 0.0

    path2 = find_path_to_root(taxonomy, node2)
    if path2: path2.append("Tatica_de_Comando")
    else: return 0.0

    # Encontra o Ancestral Comum Mais Próximo (LCS)
    lcs = None
    for n1 in path1:
        if n1 in path2:
            lcs = n1
            break
    if not lcs: return 0.0

    # Calcula N1, N2, e N3 (número de saltos/arestas)
    n1 = path1.index(lcs)
    n2 = path2.index(lcs)
    path_lcs = find_path_to_root(taxonomy, lcs)
    if path_lcs: path_lcs.append("Tatica_de_Comando")
    n3 = len(path_lcs) - 1 if path_lcs else 0
    
    denominator = n1 + n2 + (2 * n3)
    return (2 * n3) / denominator if denominator > 0 else 0.0

# --- FUNÇÕES AUXILIARES ---

def map_command_to_node(command: str):
    """Mapeia um comando de texto para um nó na nossa taxonomia (versão completa)."""
    # Simplifica o comando para a sua forma base
    cmd_base = command.strip().split(" ")[0].replace('.py', '')

    # Casos especiais
    if cmd_base == "rm" and "-rf" in command: return "rm -rf"
    if cmd_base == "kill" and "-9" in command: return "kill -9"
    if cmd_base == "rpm" and "--nodeeps" in command: return "rpm --nodeeps"
    if cmd_base == "iptables" and "-F" in command: return "iptables -F"
    if cmd_base == "rpm" and "-V" in command: return "rpm -V"
    if cmd_base == "shutdown" and "-h" in command and "now" in command: return "shutdown -h now"

    # Mapeamento direto para os outros comandos
    simple_map = [
        "dmesg", "fsck", "top", "strace", "find", "grep", "watch", "tcpdump",
        "man", "apropos", "pkill", "ps", "netstat", "ping", "dig", "gcc",
        "python", "git", "curl", "wget", "nc", "ssh"
    ]
    if cmd_base in simple_map:
        return cmd_base
        
    return None

def calculate_rule_score(similarity_scores, rule_name):
    """Calcula o score de uma regra específica baseado na média das subpersonalidades"""
    if rule_name == "Explorador_Criativo":
        score = (similarity_scores["Abertura_a_Experiencia"] + similarity_scores["Flexibilidade"]) / 2
        print(f"      🎯 {rule_name}: (Abertura({similarity_scores['Abertura_a_Experiencia']:.2f}) + Flexibilidade({similarity_scores['Flexibilidade']:.2f})) / 2 = {score:.3f}")
        return score
    
    elif rule_name == "Metodico_e_Perfeccionista":
        score = (similarity_scores["Perfeccionismo"] + similarity_scores["Paciência"]) / 2
        print(f"      🎯 {rule_name}: (Perfeccionismo({similarity_scores['Perfeccionismo']:.2f}) + Paciência({similarity_scores['Paciência']:.2f})) / 2 = {score:.3f}")
        return score
    
    elif rule_name == "Descuidado_e_Impulsivo":
        # Um score alto aqui significa alta semelhança com o descuido.
        # Medimos isso pela "dissimilaridade" com o perfeccionismo e a paciência.
        dissimilarity_perf = 1.0 - similarity_scores["Perfeccionismo"]
        dissimilarity_pac = 1.0 - similarity_scores["Paciência"]
        score = (dissimilarity_perf + dissimilarity_pac) / 2
        print(f"      🎯 {rule_name}: (Dissimilaridade_Perf({dissimilarity_perf:.2f}) + Dissimilaridade_Pac({dissimilarity_pac:.2f})) / 2 = {score:.3f}")
        return score
        
    return 0.0

def classify_profile(similarity_scores):
    """Sistema que testa todas as regras e escolhe a que mais se encaixa"""
    print(f"\n    🔍 Calculando Scores das Regras de Perfil:")
    rules = ["Explorador_Criativo", "Metodico_e_Perfeccionista", "Descuidado_e_Impulsivo"]
    rule_scores = {rule: calculate_rule_score(similarity_scores, rule) for rule in rules}
    
    # Encontra a regra com maior score
    best_rule = max(rule_scores.keys(), key=lambda k: rule_scores[k])
    best_score = rule_scores[best_rule]
    
    # Se o melhor score for muito baixo, o perfil é considerado indefinido
    if best_score < 0.35: # Limiar de confiança
        return "Indefinido"
    
    return best_rule

# Funções de estado e iptables (omitidas por brevidade)
# ...
def rule_exists(ip_address: str):
    for port in PROFILE_ACTION_PORTS.values():
        if port is None: continue
        command = ["iptables", "-t", "nat", "-C", "PREROUTING", "-s", ip_address, "-p", "tcp", "--dport", str(PORT_COWRIE1), "-j", "REDIRECT", "--to-port", str(port)]
        result = subprocess.run(command, check=False, capture_output=True)
        if result.returncode == 0: return True
    return False

def add_redirect_rules(ip_address: str, target_port: int):
    print(f"    → APLICANDO REDIRECIONAMENTO: IP {ip_address} para a porta {target_port}.")
    # Regra para tráfego vindo de fora (atacantes externos)
    prerouting_cmd = [
        "iptables", "-t", "nat", "-I", "PREROUTING", "1",
        "-s", ip_address, "-p", "tcp", "--dport", str(PORT_COWRIE1),
        "-j", "REDIRECT", "--to-port", str(target_port)
    ]
    # Regra para tráfego local (se necessário)
    output_cmd = [
        "iptables", "-t", "nat", "-A", "OUTPUT",
        "-s", ip_address, "-p", "tcp", "--dport", str(PORT_COWRIE1),
        "-j", "REDIRECT", "--to-port", str(target_port)
    ]
    print(f"      - Adicionando regra na PREROUTING...")
    subprocess.run(prerouting_cmd, check=False)
    print(f"      - Adicionando regra na OUTPUT...")
    subprocess.run(output_cmd, check=False)
    print("    → Regras aplicadas.")

def load_state():
    if not os.path.exists(STATE_FILE): return ({'inode': 0, 'position': 0}, {})
    try:
        with open(STATE_FILE, 'r') as f: state = json.load(f)
        return state.get('file_processing', {'inode': 0, 'position': 0}), state.get('session_activity', {})
    except (json.JSONDecodeError, IOError):
        return ({'inode': 0, 'position': 0}, {})

def save_state(file_state, activity_state):
    state = {'file_processing': file_state, 'session_activity': activity_state}
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2)

# --- LÓGICA PRINCIPAL ---

def main():
    if not os.path.isfile(LOGFILE): return
    file_state, session_activity = load_state()
    last_inode, last_position = file_state.get('inode', 0), file_state.get('position', 0)
    try: current_inode = os.stat(LOGFILE).st_ino
    except FileNotFoundError: return
    if current_inode != last_inode:
        last_position, session_activity = 0, {}

    with open(LOGFILE, 'r') as f:
        f.seek(last_position)
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("eventid") == "cowrie.command.input":
                    session, src_ip, command_input = log_entry.get("session"), log_entry.get("src_ip"), log_entry.get("input", "")
                    if not session or not src_ip: continue
                    command_node = map_command_to_node(command_input)
                    if command_node:
                        if session not in session_activity:
                            session_activity[session] = {'ip': src_ip, 'command_nodes': [], 'profile': None}
                        if not session_activity[session].get('profile'):
                            session_activity[session]['command_nodes'].append(command_node)
            except json.JSONDecodeError: continue
        current_position = f.tell()

    print("\n--- Análise de Logs Concluída. Iniciando Classificação de Perfis ---")
    
    for session, data in list(session_activity.items()):
        if data.get('profile'): continue
        if len(data.get('command_nodes', [])) < 3: continue

        ip, command_nodes = data['ip'], data.get('command_nodes', [])
        
        print(f"\n============================================================")
        print(f"🔍 PROCESSANDO SESSÃO: {session} | IP: {ip}")
        print(f"============================================================")
        print(f"  📝 Comandos Mapeados: {list(set(command_nodes))}")
        
        similarity_scores = {}
        print(f"\n  📊 Calculando Similaridades com Protótipos:")
        for trait, ref_node in PERSONA_PROTOTYPES.items():
            sims = [calculate_wu_palmer_similarity(COMMAND_TAXONOMY, ref_node, node) for node in command_nodes]
            average_sim = sum(sims) / len(sims) if sims else 0
            similarity_scores[trait] = average_sim
            print(f"    - Média de Similaridade com '{trait}': {average_sim:.3f}")
        
        final_profile = classify_profile(similarity_scores)
        data['profile'] = final_profile
        
        print(f"\n  ✅ PERFIL FINAL IDENTIFICADO: {final_profile}")

        target_port = PROFILE_ACTION_PORTS.get(final_profile)
        if target_port and not rule_exists(ip):
            add_redirect_rules(ip, target_port)
        elif target_port:
            print(f"    - Ação de redirecionamento para o perfil '{final_profile}' já foi aplicada para o IP {ip}.")
        else:
            print("    - Nenhuma ação de redirecionamento configurada para este perfil.")

    save_state({'inode': current_inode, 'position': current_position}, session_activity)
    print("\n--- Verificação Concluída. Estado Salvo. ---\n")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("ERRO: Este script precisa ser executado com privilégios de root (sudo) para manipular o iptables.")
    else:
        main()