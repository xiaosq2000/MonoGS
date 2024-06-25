import rich

_log_styles = {
    "Semantic-3DGS-SLAM": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold yellow",
    "Error": "bold red",
    "Map": "bold cyan",
    "Track": "bold cyan",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="Semantic-3DGS-SLAM"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
