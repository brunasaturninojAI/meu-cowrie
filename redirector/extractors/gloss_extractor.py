#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gloss_extractor.py
Extrai 'glossas' (descrições) das páginas man, similar às glossas do WordNet.
"""

import subprocess
import re
import logging
import platform

logger = logging.getLogger(__name__)

class CommandGlossExtractor:
    """Extrai 'glossas' (descrições) de forma segura.
    Em modo seguro, não executa binários externos; usa apenas fallback/local.
    """
    
    def __init__(self, safe_mode: bool = None):
        # Em Windows, habilita safe_mode por padrão para evitar abrir GUIs (ex.: gitk)
        if safe_mode is None:
            safe_mode = (platform.system().lower() == 'windows')
        self.safe_mode = safe_mode
        self.cache = {}
    
    def get_command_gloss(self, command: str) -> str:
        """Extrai descrição da página man quando seguro; caso contrário usa fallback.
        Nunca executa binários externos quando safe_mode=True.
        Enriquece com features estruturais.
        """
        if command in self.cache:
            return self.cache[command]

        # Extrai features estruturais
        structural_features = self._extract_structural_features(command)

        # Verifica primeiro fallback local (evita executar programas quando possível)
        fallback_gloss = self._get_fallback_description(command)
        if self.safe_mode:
            # Em modo seguro sempre retorna fallback (ou descrição genérica)
            enriched_gloss = f"{fallback_gloss} {structural_features}"
            self.cache[command] = enriched_gloss
            return enriched_gloss
        else:
            if fallback_gloss != f"system command: {command}":
                enriched_gloss = f"{fallback_gloss} {structural_features}"
                self.cache[command] = enriched_gloss
                return enriched_gloss

        # Processa comandos compostos
        base_cmd = command.split()[0].replace('-', '_')
        
        try:
            # Tenta obter a página man (somente em modo não seguro)
            result = subprocess.run(
                ["man", base_cmd], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                gloss = self._extract_description(result.stdout)
            else:
                # Fallback 1: Tenta --help
                gloss = self._try_help_fallback(base_cmd)
                if not gloss:
                    # Fallback 2: Dicionário predefinido
                    gloss = self._get_fallback_description(command)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, PermissionError):
            # Fallback 1: Tenta --help
            gloss = self._try_help_fallback(base_cmd)
            if not gloss:
                # Fallback 2: Dicionário predefinido
                gloss = self._get_fallback_description(command)
        
        self.cache[command] = gloss
        return gloss
    
    def _try_help_fallback(self, base_cmd: str) -> str:
        """Tenta extrair descrição usando --help; não é usado em safe_mode."""
        if self.safe_mode:
            return None
        help_options = ["--help", "-h", "-help", "help"]
        
        for help_opt in help_options:
            try:
                result = subprocess.run(
                    [base_cmd, help_opt],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                
                if result.returncode == 0 and result.stdout:
                    description = self._parse_help_output(result.stdout, base_cmd)
                    if description and len(description) > 20:  # Só aceita se for substancial
                        return description
                        
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                continue
        
        return None
    
    def _parse_help_output(self, help_text: str, cmd_name: str) -> str:
        """Extrai descrição útil do output de --help."""
        lines = help_text.split('\n')
        description_parts = []
        
        # Procura por padrões comuns de descrição
        for i, line in enumerate(lines[:10]):  # Só as primeiras 10 linhas
            line = line.strip()
            
            # Padrão 1: "comando - descrição"
            if f"{cmd_name} -" in line.lower():
                desc = line.split('-', 1)
                if len(desc) > 1:
                    description_parts.append(desc[1].strip())
            
            # Padrão 2: Primeira linha não vazia que não é usage
            elif (line and 
                  not line.startswith('Usage:') and 
                  not line.startswith('usage:') and
                  not line.startswith(cmd_name + ' ') and
                  'option' not in line.lower() and
                  len(line) > 15):
                description_parts.append(line)
                break
        
        if description_parts:
            full_desc = ' '.join(description_parts)
            # Limita o tamanho e limpa
            words = full_desc.split()[:15]  # Máximo 15 palavras
            return ' '.join(words).lower().strip('.,;')
        
        return None
    
    def _extract_description(self, man_output: str) -> str:
        """Extrai a seção DESCRIPTION da página man."""
        lines = man_output.split('\n')
        description_lines = []
        in_description = False
        
        for line in lines:
            if re.match(r'^DESCRIPTION', line.strip()):
                in_description = True
                continue
            elif re.match(r'^[A-Z][A-Z\s]+$', line.strip()) and in_description:
                break  # Próxima seção
            elif in_description and line.strip():
                description_lines.append(line.strip())
        
        if description_lines:
            return ' '.join(description_lines[:10])  # Primeiras 10 linhas
        else:
            return f"system command: {man_output.split(chr(10))[0] if man_output else 'unknown'}"
    
    def _get_fallback_description(self, command: str) -> str:
        """Descrições de fallback para comandos comuns."""
        fallbacks = {
            # Comandos originais
            "rm -rf": "remove files and directories recursively without confirmation",
            "kill -9": "terminate process using SIGKILL signal",
            "iptables -F": "flush firewall rules",
            "rpm --nodeeps": "install package ignoring dependencies",
            "watch": "execute command repeatedly",
            "tcpdump": "capture network packets",
            "dmesg": "display kernel messages",
            "fsck": "check filesystem",
            
            # Comandos modernos - Sistema e serviços (NEUTRALIZADOS)
            "systemctl": "control systemd services",
            "systemctl start": "start systemd service",
            "systemctl stop": "stop systemd service", 
            "systemctl restart": "restart systemd service",
            "systemctl status": "show systemd service status",
            "systemctl enable": "enable systemd service autostart",
            "systemctl disable": "disable systemd service autostart",
            "journalctl": "query systemd logs",
            "service": "control system services",
            
            # Container e orquestração (NEUTRALIZADOS)
            "docker": "container platform tool",
            "docker run": "create container from image",
            "docker build": "build container image",
            "docker ps": "list containers",
            "docker exec": "execute command in container",
            "docker pull": "download container image",
            "docker push": "upload container image",
            "docker-compose": "manage multi-container applications",
            "kubectl": "kubernetes command line tool",
            "kubectl apply": "apply kubernetes configuration",
            "kubectl get": "list kubernetes resources",
            "kubectl describe": "show kubernetes resource details",
            
            # Ferramentas de rede (NEUTRALIZADOS)
            "curl": "data transfer tool",
            "curl -X POST": "send HTTP POST request",
            "curl -X GET": "send HTTP GET request",
            "wget": "download files from web",
            "nmap": "network scanning tool",
            "nmap -sS": "perform TCP SYN scan",
            "nmap -sU": "perform UDP scan",
            "wireshark": "network protocol analyzer",
            "tshark": "command line network analyzer",
            "ncat": "network connection tool",
            "netcat": "network utility",
            
            # Processamento de texto (NEUTRALIZADOS)
            "awk": "text processing tool",
            "sed": "stream text editor",
            "jq": "JSON processor",
            "yq": "YAML processor",
            
            # Ferramentas de desenvolvimento (NEUTRALIZADOS)
            "git": "version control system",
            "git clone": "copy repository",
            "git commit": "save changes to repository",
            "git push": "upload changes to remote",
            "git pull": "download changes from remote",
            "npm": "node package manager",
            "yarn": "javascript package manager",
            "pip": "python package installer",
            "conda": "package manager for python",
            
            # Automação e configuração (NEUTRALIZADOS)
            "ansible": "configuration management tool",
            "ansible-playbook": "execute ansible tasks",
            "terraform": "infrastructure management tool",
            "terraform plan": "show infrastructure changes",
            "terraform apply": "apply infrastructure configuration",
            "crontab": "schedule recurring tasks",
            "crontab -e": "edit scheduled tasks",
            "at": "schedule one-time task",
            
            # Monitoramento (NEUTRALIZADOS)
            "htop": "process monitor",
            "iotop": "input output monitor",
            "nethogs": "network usage monitor",
            "ss": "socket statistics",
            "lsof": "list open files",
            "strace": "trace system calls",
            "perf": "performance analysis tool",
            
            # Segurança (NEUTRALIZADOS)
            "fail2ban": "intrusion prevention system",
            "ufw": "firewall management tool",
            "openssl": "cryptographic toolkit",
            "gpg": "encryption and signature tool",
            "sudo": "execute as different user",
            "su": "switch user",
            
            # Cloud (NEUTRALIZADOS)
            "aws": "amazon web services cli",
            "gcloud": "google cloud cli",
            "az": "azure cli",
            "helm": "kubernetes package manager",
            
            # Comandos de backup e recuperação
            "backup": "create data backup copy",
            "rsync -av": "synchronize files with backup",
            "tar -czf": "create compressed archive backup",
            "gzip": "compress files for backup",
            "zip": "create zip archive backup",
            "dump": "create filesystem backup",
            "restore": "restore from backup",
        }
        
        return fallbacks.get(command, f"system command: {command}")
    
    def clear_cache(self):
        """Limpa o cache de glossas."""
        self.cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Retorna estatísticas do cache."""
        return {
            "cached_commands": len(self.cache),
            "cache_size": sum(len(gloss) for gloss in self.cache.values())
        }

    def _extract_structural_features(self, command: str) -> str:
        """Extrai features estruturais do comando para melhorar classificação."""
        features = []
        cmd_lower = command.lower()

        # Flags destrutivas/perigosas
        dangerous_flags = ["-rf", "--force", "-f", "--no-preserve-root", "-9", "--privileged",
                          "--auto-approve", "--no-verify", "-k", "--insecure",
                          "--no-check-certificate", "-T5", "--rate="]
        for flag in dangerous_flags:
            if flag in cmd_lower:
                features.append("DANGEROUS_FLAG")
                break

        # Flags de cautela/segurança
        safe_flags = ["--check", "--dry-run", "-n", "--verify", "-i", "-p",
                     "--backup", "-v", "--verbose", "--diff", "plan"]
        for flag in safe_flags:
            if flag in cmd_lower:
                features.append("SAFE_FLAG")
                break

        # Operações destrutivas
        destructive_ops = ["rm ", "kill", "delete", "destroy", "wipe", "shred",
                          "dd if=", "mkfs", "format", "shutdown", "reboot"]
        for op in destructive_ops:
            if op in cmd_lower:
                features.append("DESTRUCTIVE_OP")
                break

        # Operações de leitura/informação
        read_ops = ["cat ", "less ", "more ", "head ", "tail ", "ls ", "ps ",
                   "top ", "status", "show", "get ", "list", "view"]
        for op in read_ops:
            if op in cmd_lower:
                features.append("READ_OP")
                break

        # Operações de rede
        network_ops = ["ssh", "scp", "curl", "wget", "nc ", "netcat", "nmap",
                      "ping", "telnet", "ftp", "http"]
        for op in network_ops:
            if op in cmd_lower:
                features.append("NETWORK_OP")
                break

        # Operações de backup/proteção
        backup_ops = ["backup", "rsync", "tar ", "zip", "dump", "stash",
                     "snapshot", "clone"]
        for op in backup_ops:
            if op in cmd_lower:
                features.append("BACKUP_OP")
                break

        # Operações privilegiadas
        priv_ops = ["sudo", "su ", "root", "chown", "chmod 777", "chmod -R 777"]
        for op in priv_ops:
            if op in cmd_lower:
                features.append("PRIVILEGED_OP")
                break

        # Redirecionamento de execução remota (muito perigoso)
        if ("curl" in cmd_lower or "wget" in cmd_lower) and ("|" in cmd_lower or "bash" in cmd_lower or "sh" in cmd_lower):
            features.append("REMOTE_EXEC")

        # Pipe para shell
        if "| bash" in cmd_lower or "| sh" in cmd_lower:
            features.append("PIPE_TO_SHELL")

        return " ".join(features) if features else ""