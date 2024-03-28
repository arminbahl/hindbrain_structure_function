def format_dict(d):
    lines = []
    for key, value in d.items():
        if isinstance(value, list):
            value = ', '.join(f"'{item}'" for item in value)
            lines.append(f"{key} = [{value}]")
        else:
            lines.append(f"{key} = '{value}'")
    return '\n'.join(lines)
