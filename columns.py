"""
Column definitions for the NSL-KDD dataset.

NSL-KDD has 41 features + 1 label column + 1 'difficulty' column (we drop difficulty).
These names come from the original KDD Cup 1999 task description, which NSL-KDD reuses.
"""

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Binary / already-0-1 columns (do NOT scale these, and be aware some are near-constant)
BINARY_COLS = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]

# Maps every specific attack name in NSL-KDD to one of 5 top-level categories.
# This mapping is the standard one used across NSL-KDD literature.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "mailbomb": "dos", "processtable": "dos", "udpstorm": "dos",
    "apache2": "dos", "worm": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L (remote-to-local)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # U2R (user-to-root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}
