import json

def update_requirements(requirements_path, outdated_packages_path):
    with open(outdated_packages_path, 'r') as f:
        outdated_packages = json.load(f)

    with open(requirements_path, 'r') as f:
        requirements_lines = f.readlines()

    outdated_map = {pkg['name']: pkg['latest_version'] for pkg in outdated_packages}

    new_requirements_lines = []
    updated_anything = False
    for line in requirements_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            new_requirements_lines.append(line)
            continue

        package_name = ''
        if '==' in line:
            package_name = line.split('==')[0]
        elif '>=' in line:
            package_name = line.split('>=')[0]
        elif '<=' in line:
            package_name = line.split('<=')[0]
        elif '~=' in line:
            package_name = line.split('~=')[0]
        elif '!=' in line:
            package_name = line.split('!=')[0]
        else: # package name only
            package_name = line.split('[')[0] if '[' in line else line


        if package_name in outdated_map:
            latest_version = outdated_map[package_name]
            if '==' in line:
                current_version_spec = line.split('==')[1]
                # Only update if the latest version is different from the pinned one
                if current_version_spec != latest_version:
                    new_requirements_lines.append(f"{package_name}=={latest_version}")
                    print(f"Updated {package_name} to {latest_version}")
                    updated_anything = True
                else:
                    new_requirements_lines.append(line) # No change needed
            else:
                # If not pinned, or other specifier, update to latest
                new_requirements_lines.append(f"{package_name}=={latest_version}")
                print(f"Updated {package_name} to {latest_version} (was {line})")
                updated_anything = True
        else:
            new_requirements_lines.append(line)

    if updated_anything:
        with open(requirements_path, 'w') as f:
            for updated_line in new_requirements_lines:
                f.write(updated_line + '\n')
        print(f"Successfully updated {requirements_path}")
    else:
        print(f"No updates needed for {requirements_path}")

if __name__ == "__main__":
    update_requirements('Fusion/LM-Fuser/requirements.txt', 'outdated_packages.json')
