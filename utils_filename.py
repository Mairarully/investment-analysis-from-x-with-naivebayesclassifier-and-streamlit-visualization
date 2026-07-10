from pathlib import Path

def generate_safe_filename(input_file, suffix, output_dir=None, ext=".csv"):
    """
    Membuat nama file output yang aman (tidak overwrite)
    """
    input_path = Path(input_file)

    base_name = input_path.stem
    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{base_name}_{suffix}{ext}"

    counter = 1
    while output_file.exists():
        counter += 1
        output_file = output_dir / f"{base_name}_{suffix}_v{counter}{ext}"

    return output_file