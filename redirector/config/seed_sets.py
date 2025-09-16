#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seed_sets.py
Conjuntos semente para classificação de comandos por traços psicológicos HEXACO.
Baseado na metodologia SentiWordNet 3.0 adaptada para as 6 dimensões HEXACO.
"""

# --- CONJUNTOS SEMENTE HEXACO (6 dimensões de personalidade) ---
SEED_SETS = {
    # ========== HONESTY-HUMILITY ==========
    "HonestyHumility_Positive": [
        # Comandos que demonstram transparência e verificação ética
        "gpg --verify", "openssl verify", "sha256sum", "md5sum", "crc32",
        "git log --verify", "rpm -V", "dpkg --verify", "checksec",
        "sudo -v", "whoami", "id", "groups", "getent", "finger",
        "history", "last", "lastlog", "w", "who", "users",
        "systemctl status", "systemctl is-active", "systemctl show",
        "docker inspect", "kubectl describe", "terraform show",
        "ansible --check", "ansible --diff", "git diff", "git status",
    ],
    
    "HonestyHumility_Negative": [
        # Comandos que demonstram manipulação e quebra de regras
        "chmod 777", "chown root", "sudo su -", "su -", "sudo -s",
        "rm -rf --no-preserve-root", "dd if=/dev/zero", "mkfs.ext4",
        "iptables -F", "iptables -P INPUT ACCEPT", "setenforce 0",
        "systemctl mask", "systemctl disable", "kill -9", "killall -9",
        "docker run --privileged", "docker run --user root", "kubectl patch --force",
        "terraform apply -auto-approve", "git push --force", "git reset --hard",
        "curl -k", "wget --no-check-certificate", "ssh -o StrictHostKeyChecking=no",
        "nc -l", "netcat -l", "socat", "proxychains", "tor",
    ],
    
    # ========== EMOTIONALITY ==========
    "Emotionality_Positive": [
        # Comandos que demonstram cautela e busca por suporte
        "backup", "rsync -av", "cp -p", "tar -czf", "gzip", "zip",
        "systemctl enable", "systemctl start", "mount -o ro", "fsck -n",
        "ping -c 1", "telnet", "ssh -v", "curl --connect-timeout",
        "nmap -sn", "nmap -sT", "traceroute", "mtr", "dig +trace",
        "tail -f", "watch", "sleep", "timeout", "wait",
        "systemctl reload", "systemctl restart", "journalctl -f",
        "docker logs -f", "kubectl logs -f", "ansible --check",
        "terraform plan", "git stash", "git branch", "git checkout -b",
    ],
    
    "Emotionality_Negative": [
        # Comandos que demonstram indiferença a riscos
        "rm -rf", "dd if=/dev/urandom", "shred -vfz", "wipe -rf",
        "kill -9", "killall -9", "pkill -f", "systemctl kill",
        "iptables -F", "ufw disable", "systemctl stop firewalld",
        "mount -o rw,remount", "chmod -R 777", "chown -R root:root",
        "docker rm -f", "docker kill", "kubectl delete --force --now",
        "terraform destroy -auto-approve", "git push --force-with-lease",
        "curl --max-time 1", "wget --timeout=1", "nmap -T5 -A",
        "nc -z", "ncat --exec", "socat -", "proxychains4 -q",
    ],
    
    # ========== EXTRAVERSION ==========
    "Extraversion_Positive": [
        # Comandos que demonstram confiança e interação social
        "ssh", "scp", "sftp", "rsync", "nc", "netcat", "socat",
        "curl -X POST", "wget --post-data", "httpie", "ab", "siege",
        "docker run -it", "docker exec -it", "kubectl exec -it",
        "ansible-playbook", "ansible all -m ping", "fabric", "chef-client",
        "git push", "git pull", "git merge", "git rebase", "git cherry-pick",
        "systemctl enable --now", "systemctl start", "service start",
        "tmux", "screen", "byobu", "nohup", "disown",
        "wall", "write", "talk", "mail", "sendmail", "mutt",
    ],
    
    "Extraversion_Negative": [
        # Comandos que demonstram introversão e isolamento
        "cat", "less", "more", "head", "tail", "grep", "awk", "sed",
        "ls", "pwd", "cd", "find", "locate", "which", "whereis",
        "vim", "nano", "emacs", "gedit", "vi", "ed",
        "make", "gcc", "g++", "python", "perl", "ruby", "node",
        "cron", "crontab -e", "at", "batch", "systemctl --user",
        "docker ps", "docker images", "kubectl get", "terraform show",
        "git log", "git show", "git blame", "git diff", "git status",
        "ps", "top", "htop", "free", "df", "du", "lsof", "netstat",
    ],
    
    # ========== AGREEABLENESS ==========
    "Agreeableness_Positive": [
        # Comandos que demonstram cooperação e flexibilidade
        "git merge", "git rebase", "git cherry-pick", "git pull --rebase",
        "systemctl reload", "systemctl restart", "service reload",
        "docker-compose up", "docker-compose restart", "kubectl apply",
        "ansible-playbook --check", "terraform plan", "helm upgrade",
        "chmod +x", "chown user:group", "setfacl -m", "usermod -aG",
        "mount", "umount", "sync", "fsync", "fdisk -l",
        "cp -i", "mv -i", "rm -i", "ln -s", "rsync -av",
        "curl --retry", "wget --retry-connrefused", "ping -c 3",
    ],
    
    "Agreeableness_Negative": [
        # Comandos que demonstram rigidez e confronto
        "kill -9", "killall -9", "pkill -KILL", "systemctl kill",
        "iptables -j REJECT", "iptables -j DROP", "ufw deny",
        "chmod 000", "chattr +i", "mount -o ro", "umount -f",
        "git push --force", "git reset --hard", "git clean -fd",
        "docker rm -f", "docker kill", "kubectl delete --force",
        "terraform destroy", "helm uninstall", "systemctl mask",
        "dd if=/dev/zero", "shred -vfz", "wipe -rf", "secure-delete",
        "nmap -sS -T5", "nmap -A", "masscan", "zmap", "unicornscan",
    ],
    
    # ========== CONSCIENTIOUSNESS ==========
    "Conscientiousness_Positive": [
        # Comandos que demonstram organização e disciplina
        "systemctl enable", "systemctl daemon-reload", "systemctl status",
        "crontab -e", "at", "batch", "systemd-run", "timedatectl",
        "backup", "rsync -av --progress", "tar -czf", "gzip", "bzip2",
        "fsck", "e2fsck", "badblocks", "smartctl", "hdparm",
        "rpm -V", "dpkg --verify", "checksec", "rkhunter", "chkrootkit",
        "git add", "git commit -m", "git tag", "git push origin",
        "docker build", "docker-compose build", "kubectl apply -f",
        "terraform plan", "terraform apply", "ansible-playbook --check",
        "make clean", "make install", "configure --prefix", "cmake",
    ],
    
    "Conscientiousness_Negative": [
        # Comandos que demonstram desorganização e impulsividade
        "rm -rf", "dd if=/dev/urandom", "kill -9", "killall",
        "shutdown -h now", "reboot", "halt", "poweroff",
        "iptables -F", "systemctl --force", "systemctl mask",
        "chmod 777", "chown -R", "mount -o rw,remount",
        "docker run --rm", "docker kill", "kubectl delete --all",
        "terraform destroy -auto-approve", "git push --force",
        "curl -k", "wget --no-check-certificate", "ssh -o StrictHostKeyChecking=no",
        "nmap -T5", "masscan --rate=1000", "nc -l -p", "socat -",
        "python -c", "bash -c", "eval", "exec", "source",
    ],
    
    # ========== OPENNESS TO EXPERIENCE ==========
    "OpennessToExperience_Positive": [
        # Comandos que demonstram curiosidade e criatividade
        "curl -X POST", "wget --post-data", "httpie POST", "jq", "yq",
        "python -c", "perl -e", "ruby -e", "node -e", "bash -c",
        "awk", "sed", "tr", "cut", "sort", "uniq", "tee", "xargs",
        "find", "locate", "grep -r", "ag", "rg", "ripgrep",
        "strace", "ltrace", "gdb", "valgrind", "perf", "objdump",
        "nmap --script", "ncat --exec", "socat", "proxychains",
        "docker-compose", "kubectl create", "helm create", "terraform init",
        "ansible-galaxy", "vagrant", "packer", "consul", "vault",
        "tmux", "screen", "vim", "emacs", "git", "svn", "hg",
    ],
    
    "OpennessToExperience_Negative": [
        # Comandos básicos e convencionais
        "cat", "ls", "pwd", "cd", "echo", "printf", "date", "whoami",
        "cp", "mv", "mkdir", "rmdir", "touch", "chmod", "chown",
        "ps", "top", "free", "df", "du", "uptime", "w", "who",
        "systemctl list-units", "systemctl is-active", "systemctl show",
        "docker ps", "docker images", "docker version", "docker info",
        "kubectl get pods", "kubectl get nodes", "kubectl version",
        "git status", "git log", "git show", "git diff", "git pull",
        "terraform show", "terraform version", "ansible --version",
        "curl -O", "wget", "ping", "traceroute", "netstat", "ss",
    ]
} 