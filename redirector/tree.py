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
# Uso:          sudo python3 tree.py
# -----------------------------------------------------------------

import json
import subprocess
import os

# --- PAINEL DE CONTROLE E CONFIGURAÇÃO ---

# Configuração dos arquivos de log para cada honeypot
HONEYPOT_LOGS = {
    'cowrie1': "/home/brunasaturnino/meu-cowrie/cowrie1/var/log/cowrie/cowrie.json",
    'cowrie2': "/home/brunasaturnino/meu-cowrie/cowrie2/var/log/cowrie/cowrie.json",
    'cowrie3': "/home/brunasaturnino/meu-cowrie/cowrie3/var/log/cowrie/cowrie.json",
    'cowrie4': "/home/brunasaturnino/meu-cowrie/cowrie4/var/log/cowrie/cowrie.json",
    'cowrie5': "/home/brunasaturnino/meu-cowrie/cowrie5/var/log/cowrie/cowrie.json"
}

# Mapeamento de portas para honeypots
PORT_TO_HONEYPOT = {
    2222: 'cowrie1',  # Honeypot inicial
    2224: 'cowrie2',  # Descuidado_e_Impulsivo
    2226: 'cowrie3',  # Metodico_e_Perfeccionista
    2228: 'cowrie4'   # Explorador_Criativo
}

HONEYPOT_TO_PORT = {v: k for k, v in PORT_TO_HONEYPOT.items()}

PORT_COWRIE1 = 2222  # Mantido para compatibilidade

STATE_FILE = "comprehensive_profiler.state"

# 1. AÇÕES: Mapeamento dos Perfis Finais para as Portas de Destino
PROFILE_ACTION_PORTS = {
    "Descuidado_e_Impulsivo": 2224,
    "Metodico_e_Perfeccionista": 2226,
    "Explorador_Criativo": 2228,
    "Indefinido": None
}

# 2. PROTÓTIPOS: Nós da taxonomia que melhor representam cada traço de personalidade
PERSONA_PROTOTYPES = {
    "Perfeccionismo": "dmesg",
    "Paciência": "watch",
    "Flexibilidade": "man",
    "Abertura_a_Experiencia": "gcc"
}

# 3. CONHECIMENTO: A Taxonomia Hierárquica de Comandos
COMMAND_TAXONOMY = {
    "Tatica_de_Comando": { # Raiz
        "Foco_em_Precisao": { # Traço: Perfeccionismo
            "Alta_Precisao": {
                "Verificacao_Sistema": {
                    # Alta precisão – verificação e diagnóstico
                    "dmesg": {}, "fsck": {}, "e2fsck": {}, "dosfsck": {}, "badblocks": {},
                    "df": {}, "getconf": {}, "lscpu": {}, "ethtool": {}, "uname": {},
                    "rpm -V": {}, "rpm --checksig": {}, "dpkg -s": {}, "dpkg -L": {},
                    "apt-get check": {}
                },
                "Analise_Detalhada": {
                    # Alta precisão – análise contínua/monitoramento
                    "top": {}, "ps": {}, "lsof": {}, "strace": {}, "watch": {}, "tail -f": {},
                    "tcpdump": {}, "netstat": {}, "diff": {}, "find": {}, "grep": {}
                }
            },
            "Baixa_Precisao": {
                "Acao_Forcada": {
                    # Baixa precisão – ações destrutivas ou sem checagem
                    "rm -rf": {}, "kill -9": {}, "rpm -ivh --nodeeps": {}, "nohup": {}
                },
                "Ignorar_Procedimentos": {
                    "rpm --nodeeps": {}, "iptables -F": {}, "iptables -t nat -F": {}, "iptables -t filter -F": {}
                }
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

# --- MOTOR DE CÁLCULO: WU & PALMER  ---

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

# --------- PRECISION HELPERS ---------
def precision_level(node_name: str):
    """Return 'high', 'low', or None depending on where the node lives inside Foco_em_Precisao."""
    path = find_path_to_root(COMMAND_TAXONOMY, node_name) or []
    if "Alta_Precisao" in path:
        return "high"
    if "Baixa_Precisao" in path:
        return "low"
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

# Funções de estado e iptables 
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

def remove_redirect_rules(ip_address: str, port: int):
    """Remove regras de redirecionamento específicas para um IP e porta."""
    print(f"    → REMOVENDO REDIRECIONAMENTO: IP {ip_address} da porta {port}.")
    # Remove regra da PREROUTING
    prerouting_cmd = [
        "iptables", "-t", "nat", "-D", "PREROUTING",
        "-s", ip_address, "-p", "tcp", "--dport", str(PORT_COWRIE1),
        "-j", "REDIRECT", "--to-port", str(port)
    ]
    # Remove regra da OUTPUT
    output_cmd = [
        "iptables", "-t", "nat", "-D", "OUTPUT",
        "-s", ip_address, "-p", "tcp", "--dport", str(PORT_COWRIE1),
        "-j", "REDIRECT", "--to-port", str(port)
    ]
    print(f"      - Removendo regra da PREROUTING...")
    subprocess.run(prerouting_cmd, check=False)
    print(f"      - Removendo regra da OUTPUT...")
    subprocess.run(output_cmd, check=False)
    print("    → Regras removidas.")

def load_state():
    if not os.path.exists(STATE_FILE): 
        return {
            'file_processing': {honeypot: {'inode': 0, 'position': 0} for honeypot in HONEYPOT_LOGS.keys()},
            'session_activity': {},
            'ip_analysis_state': {}
        }
    try:
        with open(STATE_FILE, 'r') as f: 
            state = json.load(f)
            # Garante que todos os honeypots estejam no estado
            if 'file_processing' not in state:
                state['file_processing'] = {}
            for honeypot in HONEYPOT_LOGS.keys():
                if honeypot not in state['file_processing']:
                    state['file_processing'][honeypot] = {'inode': 0, 'position': 0}
            return state
    except (json.JSONDecodeError, IOError):
        return {
            'file_processing': {honeypot: {'inode': 0, 'position': 0} for honeypot in HONEYPOT_LOGS.keys()},
            'session_activity': {},
            'ip_analysis_state': {}
        }

def save_state(file_states, activity_state, ip_state):
    state = {
        'file_processing': file_states,
        'session_activity': activity_state,
        'ip_analysis_state': ip_state
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def process_log_file(logfile, last_position, session_activity, ip_analysis_state):
    """Processa um arquivo de log específico e retorna a nova posição."""
    if not os.path.isfile(logfile):
        return last_position, session_activity

    with open(logfile, 'r') as f:
        f.seek(last_position)
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("eventid") == "cowrie.command.input":
                    session = log_entry.get("session")
                    src_ip = log_entry.get("src_ip")
                    command_input = log_entry.get("input", "")
                    if not session or not src_ip: 
                        continue
                    
                    command_node = map_command_to_node(command_input)
                    if command_node:
                        if session not in session_activity:
                            session_activity[session] = {
                                'ip': src_ip, 
                                'command_nodes': [], 
                                'profile': None,
                                'honeypot': None  # Adicionado para rastrear em qual honeypot a sessão está
                            }
                        session_activity[session]['command_nodes'].append(command_node)
                        
                        if src_ip not in ip_analysis_state:
                            ip_analysis_state[src_ip] = {
                                'last_analyzed_cmd_count': 0,
                                'current_profile': None,
                                'current_honeypot': None  # Adicionado para rastrear o honeypot atual
                            }
            except json.JSONDecodeError:
                continue
        return f.tell(), session_activity

def get_all_commands_by_ip(session_activity, target_ip):
    """Retorna todos os comandos executados por um IP em todas as suas sessões."""
    all_commands = []
    for session_data in session_activity.values():
        if session_data['ip'] == target_ip:
            all_commands.extend(session_data.get('command_nodes', []))
    return all_commands

def main():
    # Carrega o estado para todos os honeypots
    state_data = load_state()
    file_states = state_data['file_processing']
    session_activity = state_data.get('session_activity', {})
    ip_analysis_state = state_data.get('ip_analysis_state', {})

    # Processa logs de todos os honeypots
    for honeypot, logfile in HONEYPOT_LOGS.items():
        try:
            current_inode = os.stat(logfile).st_ino
        except FileNotFoundError:
            continue

        # Reseta o estado se o arquivo mudou
        if current_inode != file_states[honeypot].get('inode', 0):
            file_states[honeypot] = {'inode': current_inode, 'position': 0}

        # Processa o arquivo de log
        new_position, session_activity = process_log_file(
            logfile,
            file_states[honeypot]['position'],
            session_activity,
            ip_analysis_state
        )
        
        # Atualiza a posição no arquivo
        file_states[honeypot]['position'] = new_position

    print("\n--- Análise de Logs Concluída. Iniciando Classificação de Perfis ---")
    
    # Agrupa sessões por IP para análise
    ips_to_analyze = set(data['ip'] for data in session_activity.values())
    
    for ip in ips_to_analyze:
        # Obtém todos os comandos deste IP em todas as sessões de todos os honeypots
        all_commands = get_all_commands_by_ip(session_activity, ip)
        
        if len(all_commands) < 3: 
            continue
        
        current_cmd_count = len(all_commands)
        # Garante que o IP exista no dicionário de estado
        if ip not in ip_analysis_state:
            ip_analysis_state[ip] = {
                'last_analyzed_cmd_count': 0,
                'current_profile': None,
                'current_honeypot': None
            }

        if current_cmd_count == ip_analysis_state[ip]['last_analyzed_cmd_count']:
            print(f"\n  ✅ PERFIL MANTIDO (sem novos comandos): {ip_analysis_state[ip]['current_profile']}")
            print(f"  🎯 HONEYPOT: {ip_analysis_state[ip]['current_honeypot']}")
            continue

        print(f"\n============================================================")
        print(f"🔍 PROCESSANDO IP: {ip}")
        if ip_analysis_state[ip]['current_profile']:
            print(f"📌 Perfil Anterior: {ip_analysis_state[ip]['current_profile']}")
            print(f"🎯 Honeypot Atual: {ip_analysis_state[ip]['current_honeypot']}")
        print(f"============================================================")
        print(f"  📝 Total de Comandos Mapeados: {len(all_commands)}")
        print(f"  📝 Comandos Únicos: {list(set(all_commands))}")
        
        similarity_scores = {}
        print(f"\n  📊 Calculando Similaridades com Protótipos:")
        for trait, ref_node in PERSONA_PROTOTYPES.items():
            sims = []
            for node in all_commands:
                sim = calculate_wu_palmer_similarity(COMMAND_TAXONOMY, ref_node, node)
                if trait == "Perfeccionismo":
                    prec_level = precision_level(node)
                    if prec_level == "low":
                        sim *= -1  # penaliza comandos de baixa precisão
                    elif prec_level is None:
                        continue  # neutro, fora do eixo de precisão
                sims.append(sim)

            average_sim = sum(sims) / len(sims) if sims else 0
            similarity_scores[trait] = average_sim
            print(f"    - Média de Similaridade com '{trait}': {average_sim:.3f}")
        
        new_profile = classify_profile(similarity_scores)
        old_profile = ip_analysis_state[ip]['current_profile']
        
        # Atualiza o contador de comandos analisados para este IP
        ip_analysis_state[ip]['last_analyzed_cmd_count'] = current_cmd_count
        
        # Se o perfil mudou, atualiza e aplica novas regras
        if new_profile != old_profile:
            ip_analysis_state[ip]['current_profile'] = new_profile
            
            # Determina o novo honeypot com base no perfil
            target_port = PROFILE_ACTION_PORTS.get(new_profile)
            new_honeypot = PORT_TO_HONEYPOT.get(target_port) if target_port else None
            
            print(f"\n  🔄 PERFIL ATUALIZADO: {old_profile if old_profile else 'Nenhum'} -> {new_profile}")
            print(f"  🔄 HONEYPOT: {ip_analysis_state[ip]['current_honeypot'] if ip_analysis_state[ip]['current_honeypot'] else 'Nenhum'} -> {new_honeypot if new_honeypot else 'Nenhum'}")
            
            # Atualiza o honeypot atual no estado do IP
            ip_analysis_state[ip]['current_honeypot'] = new_honeypot
            
            # Atualiza o perfil em todas as sessões deste IP
            for session_data in session_activity.values():
                if session_data['ip'] == ip:
                    session_data['profile'] = new_profile
                    session_data['honeypot'] = new_honeypot
            
            # Remove regras antigas se existirem
            if old_profile and PROFILE_ACTION_PORTS.get(old_profile):
                old_port = PROFILE_ACTION_PORTS.get(old_profile)
                if old_port is not None:
                    remove_redirect_rules(ip, old_port)
            
            # Aplica novas regras
            if target_port and not rule_exists(ip):
                add_redirect_rules(ip, target_port)
            elif target_port:
                print(f"    - Ação de redirecionamento para o perfil '{new_profile}' já foi aplicada para o IP {ip}.")
            else:
                print("    - Nenhuma ação de redirecionamento configurada para este perfil.")
        else:
            print(f"\n  ✅ PERFIL MANTIDO: {new_profile}")
            print(f"  🎯 HONEYPOT: {ip_analysis_state[ip]['current_honeypot']}")

    # Atualiza o estado com as informações das sessões
    save_state(file_states, session_activity, ip_analysis_state)
    print("\n--- Verificação Concluída. Estado Salvo. ---\n")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("ERRO: Este script precisa ser executado com privilégios de root (sudo) para manipular o iptables.")
    else:
        main()