#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
# Script:       score.py
# Finalidade:
#   - Versão FINAL com persistência de pontuação entre execuções.
#   - Salva a posição do log E o placar de cada sessão no arquivo de estado.
#
# Uso:          sudo python3 score.py
# -----------------------------------------------------------------

import json
import subprocess
import os

# --- CONFIGURAÇÃO ---
LOGFILE = "/home/brunasaturnino/meu-cowrie/cowrie1/var/log/cowrie/cowrie.json"
STATE_FILE = "log_processor.state"
PORT1 = "2222"
PORT2 = "2224"
SCORE_THRESHOLD = 5

# --- PONTUAÇÃO DOS COMANDOS ---
HIGH_SKILL_SCORE = 2
HIGH_SKILL_COMMANDS = [
    "man", "--help", "apropos", "whatis", "strace", "tcpdump", "watch",
    "rsync", "dump", "restore", "sed", "grep -E", "grep -P",
]
LOW_SKILL_SCORE = 1
LOW_SKILL_COMMANDS = [
    "rm -rf", "rpm -ivh --nodeps", "iptables -t filter -F", "kill -9",
]

# -----------------------------------------------------------------
# FUNÇÕES DE ESTADO
# -----------------------------------------------------------------

def load_state():
    """Lê o estado completo: posição do arquivo E os placares das sessões."""
    default_file_state = {'inode': 0, 'position': 0}
    default_scores_state = {}
    
    if not os.path.exists(STATE_FILE):
        return default_file_state, default_scores_state
    
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            file_state = state.get('file_processing', default_file_state)
            scores_state = state.get('session_scores', default_scores_state)
            return file_state, scores_state
    except (json.JSONDecodeError, IOError):
        return default_file_state, default_scores_state

def save_state(file_state, scores_state):
    """Salva o estado completo no STATE_FILE."""
    state = {
        'file_processing': file_state,
        'session_scores': scores_state
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2) # indent=2 para facilitar a leitura manual

# -----------------------------------------------------------------
# FUNÇÕES AUXILIARES 
# -----------------------------------------------------------------
def get_command_score(command: str) -> int:
    for pattern in HIGH_SKILL_COMMANDS:
        if pattern in command: return HIGH_SKILL_SCORE
    for pattern in LOW_SKILL_COMMANDS:
        if pattern in command: return LOW_SKILL_SCORE
    return 0

def rule_exists_for_ip(ip_address: str) -> bool:
    command = ["iptables", "-t", "nat", "-C", "PREROUTING", "-p", "tcp", "-s", ip_address, "--dport", PORT1, "-j", "REDIRECT", "--to-port", PORT2]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("ERRO: O comando 'iptables' não foi encontrado.")
        return True

def add_redirect_rules(ip_address: str):
    print(f" MÁXIMO ATINGIDO ({ip_address})! Adicionando regras de redirecionamento...")
    commands_to_run = [
        ["iptables", "-t", "nat", "-I", "PREROUTING", "1", "-p", "tcp", "-s", ip_address, "--dport", PORT1, "-j", "REDIRECT", "--to-port", PORT2],
        ["iptables", "-t", "nat", "-A", "OUTPUT", "-p", "tcp", "-d", "127.0.0.1", "-s", ip_address, "--dport", PORT1, "-j", "REDIRECT", "--to-port", PORT2]
    ]
    print(f" → Adicionando regras para {ip_address} ({PORT1}→{PORT2}).")
    for cmd in commands_to_run:
        subprocess.run(cmd, check=False)
    print(f"   Regras para {ip_address} adicionadas.")

# -----------------------------------------------------------------
# SCRIPT PRINCIPAL 
# -----------------------------------------------------------------
def main():
    if not os.path.isfile(LOGFILE):
        print(f"ERRO: Logfile não encontrado em '{LOGFILE}'.")
        return

    # 1. Carrega o estado da última execução (posição E placares)
    file_state, session_scores = load_state()
    
    last_inode = file_state.get('inode', 0)
    last_position = file_state.get('position', 0)
    
    try:
        current_inode = os.stat(LOGFILE).st_ino
    except FileNotFoundError:
        return

    if current_inode != last_inode:
        print("Detectada rotação de log. Reiniciando placares e posição.")
        last_position = 0
        session_scores = {} # Zera os placares pois o log é novo
    
    print(f"Iniciando análise. Posição inicial: {last_position}. Sessões ativas: {len(session_scores)}")
    
    current_position = last_position
    with open(LOGFILE, 'r') as f:
        f.seek(last_position)
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("eventid") == "cowrie.command.input":
                    session = log_entry.get("session")
                    src_ip = log_entry.get("src_ip")
                    command_input = log_entry.get("input", "")
                    
                    if not session or not src_ip: continue
                    
                    # Atualiza o placar da sessão com a nova pontuação
                    score = get_command_score(command_input)
                    if score != 0:
                        # Se a sessão não existe no placar, cria um dicionário para ela
                        if session not in session_scores:
                            session_scores[session] = {'ip': src_ip, 'score': 0}
                        # Acumula a pontuação
                        session_scores[session]['score'] += score
            except json.JSONDecodeError:
                continue
        current_position = f.tell()

    print("Análise de novas linhas concluída. Verificando placares acumulados...")
    
    # Cria uma cópia da lista de chaves para poder remover itens do dicionário enquanto itera
    for session in list(session_scores.keys()):
        data = session_scores[session]
        ip = data['ip']
        score = data['score']
        
        print(f"Sessão: {session} | IP: {ip} | Pontuação Final: {score}")

        if score >= SCORE_THRESHOLD:
            if not rule_exists_for_ip(ip):
                add_redirect_rules(ip)
            
            # Uma vez que o atacante foi redirecionado, removemos ele do placar
            # para não poluir o arquivo de estado eternamente.
            del session_scores[session]
            print(f" → Sessão {session} redirecionada e removida do placar ativo.")

    # 7. Salva o estado atualizado para a próxima execução
    new_file_state = {'inode': current_inode, 'position': current_position}
    save_state(new_file_state, session_scores)
    print(f"Verificação concluída. Progresso e placares salvos.")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("ERRO: Execute com sudo para manipular o iptables.")
    else:
        main()