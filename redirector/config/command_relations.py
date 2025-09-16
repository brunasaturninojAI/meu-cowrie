#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
command_relations.py
Relações taxonômicas entre comandos para expansão semi-supervisionada.
Baseado nas relações do WordNet adaptadas para comandos Unix/Linux e dimensões HEXACO.
"""

# --- RELAÇÕES TAXONÔMICAS EXPANDIDAS PARA HEXACO (6 dimensões) ---
COMMAND_RELATIONS = {
    # Funcionalidade similar (como "similar-to" do WordNet)
    "similar": {
        "ps": ["top", "htop", "pstree", "pgrep", "pidof", "atop", "iotop"],
        "ls": ["dir", "ll", "tree", "find", "locate", "stat", "file"],
        "cat": ["more", "less", "head", "tail", "tac", "nl", "pr"],
        "grep": ["egrep", "fgrep", "ag", "rg", "ack", "ripgrep", "pcregrep"],
        "vi": ["vim", "nano", "emacs", "gedit", "kate", "micro", "joe"],
        "rm": ["rmdir", "unlink", "shred", "wipe", "secure-delete", "srm"],
        "cp": ["mv", "rsync", "dd", "tar", "cpio", "pax", "install"],
        "chmod": ["chown", "chgrp", "setfacl", "getfacl", "umask", "chattr"],
        "mount": ["umount", "mountpoint", "findmnt", "blkid", "lsblk", "df"],
        "ping": ["traceroute", "mtr", "fping", "nping", "hping3", "arping"],
        "netstat": ["ss", "lsof", "fuser", "netstat", "ip", "iftop", "nethogs"],
        "gcc": ["g++", "clang", "clang++", "icc", "pgcc", "tcc"],
        
        # Comandos de segurança e verificação (HEXACO: Honesty-Humility)
        "gpg": ["openssl", "age", "gnupg2", "gpg2", "pgp"],
        "sha256sum": ["md5sum", "sha1sum", "sha512sum", "crc32", "cksum"],
        "sudo": ["su", "doas", "pkexec", "runuser", "setuid"],
        
        # Comandos modernos similares
        "systemctl": ["service", "chkconfig", "update-rc.d", "rc-service", "sysv-rc-conf"],
        "docker": ["podman", "containerd", "runc", "buildah", "skopeo"],
        "kubectl": ["helm", "kustomize", "oc", "docker-compose", "kompose"],
        "curl": ["wget", "httpie", "aria2c", "axel", "lynx"],
        "nmap": ["masscan", "zmap", "unicornscan", "hping3", "nping"],
        "wireshark": ["tshark", "tcpdump", "dumpcap", "ettercap", "ngrep"],
        "git": ["svn", "hg", "bzr", "fossil", "darcs", "monotone"],
        "ansible": ["puppet", "chef", "saltstack", "fabric", "capistrano"],
        "terraform": ["pulumi", "cloudformation", "helm", "kustomize", "crossplane"],
        "awk": ["sed", "cut", "tr", "sort", "uniq", "column"],
        "jq": ["yq", "xmllint", "csvkit", "miller", "fx"],
        "tmux": ["screen", "byobu", "zellij", "dvtm", "abduco"],
    },
    
    # Funcionalidade oposta (como "antonym" do WordNet)
    "antonym": {
        "start": ["stop", "kill", "terminate", "end", "halt", "pause"],
        "mount": ["umount", "unmount", "dismount", "eject"],
        "compress": ["decompress", "uncompress", "extract", "unzip", "untar", "expand"],
        "encrypt": ["decrypt", "decipher", "decode", "unlock"],
        "connect": ["disconnect", "close", "shutdown", "exit", "logout"],
        "enable": ["disable", "block", "prevent", "deny", "mask"],
        "install": ["uninstall", "remove", "purge", "erase", "delete"],
        "create": ["delete", "destroy", "remove", "erase", "wipe"],
        "lock": ["unlock", "open", "free", "release"],
        "allow": ["deny", "block", "reject", "drop", "refuse"],
        
        # Comandos modernos opostos
        "systemctl start": ["systemctl stop", "systemctl disable", "systemctl kill"],
        "systemctl enable": ["systemctl disable", "systemctl mask", "systemctl stop"],
        "docker run": ["docker stop", "docker kill", "docker rm", "docker pause"],
        "docker pull": ["docker rmi", "docker image prune", "docker system prune"],
        "kubectl apply": ["kubectl delete", "kubectl patch", "kubectl rollback"],
        "terraform apply": ["terraform destroy", "terraform plan", "terraform refresh"],
        "git push": ["git pull", "git fetch", "git reset"],
        "curl -X POST": ["curl -X DELETE", "curl -X GET", "curl -X PUT"],
        "chmod +x": ["chmod -x", "chmod 000", "chattr +i"],
        "mount -o rw": ["mount -o ro", "umount", "mount -o noexec"],
    },
    
    # Relações de ferramentas (como "derived-from" do WordNet)
    "derived_from": {
        "gcc": ["g++", "gdb", "gprof", "gcov", "gcc-ar", "gcc-nm"],
        "git": ["gitk", "git-log", "git-gui", "git-citool", "git-cola"],
        "docker": ["docker-compose", "docker-machine", "docker-swarm", "dockerfile"],
        "ssh": ["scp", "sftp", "ssh-keygen", "ssh-agent", "ssh-add", "ssh-copy-id"],
        "python": ["pip", "conda", "poetry", "pipenv", "pyenv", "virtualenv"],
        "node": ["npm", "yarn", "pnpm", "npx", "node-gyp", "nvm"],
        "vim": ["gvim", "vimdiff", "vimtutor", "xxd", "view"],
        "openssl": ["openssl-req", "openssl-x509", "openssl-rsa", "openssl-dgst"],
        
        # Ferramentas modernas derivadas
        "systemctl": ["journalctl", "systemd-analyze", "loginctl", "timedatectl", "hostnamectl"],
        "docker": ["docker-compose", "dockerfile", "docker-machine", "docker-swarm", "docker-buildx"],
        "kubernetes": ["kubectl", "kubeadm", "kubelet", "kube-proxy", "kube-scheduler"],
        "ansible": ["ansible-playbook", "ansible-vault", "ansible-galaxy", "ansible-config"],
        "terraform": ["terraform-docs", "terragrunt", "tflint", "checkov", "terrascan"],
        "nmap": ["ncat", "nping", "ndiff", "zenmap", "nmap-parse"],
        "wireshark": ["tshark", "dumpcap", "editcap", "mergecap", "capinfos"],
        "curl": ["curl-config", "httpie", "wget", "aria2c"],
        "gpg": ["gpg-agent", "gpg-connect-agent", "gpgconf", "gpgsm"],
    },
    
    # Categorias funcionais expandidas para HEXACO (como "also-see" do WordNet)
    "also_see": {
        # Honesty-Humility relacionados
        "integrity_verification": ["gpg --verify", "openssl verify", "sha256sum", "md5sum", "checksec", "rpm -V", "dpkg --verify", "rkhunter", "chkrootkit", "lynis"],
        "transparency_tools": ["history", "last", "lastlog", "w", "who", "users", "finger", "id", "groups", "getent"],
        "ethical_access": ["sudo -v", "sudo -l", "whoami", "id", "groups", "umask", "getfacl", "setfacl"],
        
        # Emotionality relacionados
        "cautious_operations": ["cp -i", "mv -i", "rm -i", "mount -o ro", "fsck -n", "rsync --dry-run", "git stash"],
        "backup_recovery": ["backup", "rsync -av", "tar -czf", "gzip", "zip", "dump", "restore", "dd conv=sync"],
        "monitoring_watching": ["tail -f", "watch", "journalctl -f", "docker logs -f", "kubectl logs -f", "tcpdump"],
        
        # Extraversion relacionados
        "network_communication": ["ssh", "scp", "sftp", "rsync", "nc", "netcat", "socat", "telnet", "ftp"],
        "interactive_tools": ["tmux", "screen", "byobu", "docker exec -it", "kubectl exec -it", "ansible-playbook"],
        "collaboration": ["git push", "git pull", "git merge", "git rebase", "wall", "write", "talk", "mail"],
        
        # Agreeableness relacionados
        "cooperative_operations": ["git merge", "git rebase", "systemctl reload", "docker-compose restart", "kubectl apply"],
        "gentle_commands": ["curl --retry", "wget --retry-connrefused", "ping -c 3", "systemctl try-restart"],
        "flexible_tools": ["chmod +x", "chown user:group", "usermod -aG", "ln -s", "mount", "umount"],
        
        # Conscientiousness relacionados
        "organized_scheduling": ["crontab -e", "at", "batch", "systemd-run", "timedatectl", "systemctl enable"],
        "systematic_building": ["make", "cmake", "configure", "docker build", "terraform plan", "ansible-playbook --check"],
        "quality_control": ["fsck", "e2fsck", "badblocks", "smartctl", "git add", "git commit -m", "git tag"],
        
        # Openness to Experience relacionados
        "creative_scripting": ["python -c", "perl -e", "ruby -e", "bash -c", "awk", "sed", "jq", "yq"],
        "exploration_tools": ["find", "locate", "grep -r", "ag", "rg", "strace", "ltrace", "gdb", "perf"],
        "innovative_tools": ["docker-compose", "kubectl create", "helm create", "terraform init", "vagrant", "packer"],
        
        # Categorias técnicas tradicionais
        "network_analysis": ["netstat", "ss", "lsof", "tcpdump", "wireshark", "nmap", "ncat", "netcat", "tshark", "iftop", "nethogs"],
        "process_monitoring": ["ps", "top", "htop", "pstree", "pgrep", "pidof", "kill", "killall", "iotop", "atop"],
        "file_operations": ["ls", "find", "locate", "which", "whereis", "file", "stat", "touch", "tree", "fd"],
        "system_info": ["uname", "lscpu", "lsblk", "df", "du", "free", "uptime", "who", "w", "id"],
        "text_processing": ["grep", "sed", "awk", "cut", "paste", "join", "sort", "uniq", "tr", "wc"],
        "compression": ["gzip", "bzip2", "xz", "zip", "tar", "7z", "rar", "lzma", "zstd", "lz4"],
        "security": ["chmod", "chown", "umask", "setfacl", "getfacl", "passwd", "su", "sudo", "gpg", "openssl"],
        "development": ["gcc", "make", "cmake", "git", "vim", "emacs", "gdb", "valgrind", "strace", "ltrace"],
        
        # Categorias modernas
        "container_management": ["docker", "docker-compose", "podman", "buildah", "skopeo", "runc", "containerd"],
        "kubernetes_tools": ["kubectl", "helm", "kustomize", "kubeadm", "kubelet", "kube-proxy", "k9s"],
        "service_management": ["systemctl", "service", "journalctl", "systemd-analyze", "chkconfig", "rc-service"],
        "cloud_tools": ["aws", "gcloud", "az", "terraform", "pulumi", "cloudformation", "sam"],
        "automation": ["ansible", "puppet", "chef", "saltstack", "fabric", "terraform", "vagrant"],
        "monitoring_advanced": ["htop", "iotop", "nethogs", "iftop", "atop", "perf", "strace", "prometheus"],
        "network_security": ["nmap", "wireshark", "tshark", "tcpdump", "ncat", "socat", "iptables", "ufw"],
        "data_processing": ["jq", "yq", "xmllint", "csvkit", "miller", "awk", "sed", "pandas"],
        "package_management": ["npm", "yarn", "pip", "conda", "apt", "yum", "dnf", "pacman"],
        "version_control": ["git", "svn", "hg", "bzr", "fossil", "perforce", "darcs"],
        "web_tools": ["curl", "wget", "httpie", "ab", "siege", "wrk", "hey"],
        "shell_enhancement": ["tmux", "screen", "zsh", "fish", "starship", "exa", "bat", "fd"],
    }
} 