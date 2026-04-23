from mkdocs_gen_files import open as open_file
from pathlib import Path
import pkgutil
import ppkt2synergy

package = "ppkt2synergy"
exclude_modules = {"synergy_tree"}
modules = []

for module in pkgutil.walk_packages(ppkt2synergy.__path__):
    if module.name in exclude_modules:
        continue

    mod_name = f"{package}.{module.name}"
    modules.append(module.name)

    path = Path("api", f"{module.name}.md")
    with open_file(path, "w") as f:
        print(f"# {mod_name}\n", file=f)
        print(f"::: {mod_name}", file=f)

with open_file("api/index.md", "w") as f:
    f.write("# API Reference\n\n")
    f.write("Automatically generated API documentation.\n\n")
    for m in modules:
        f.write(f"- [{m}]({m}.md)\n")