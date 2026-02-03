# run.py à la racine du projet
#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline d'entraînement et d'évaluation
"""

import os
import sys
import argparse
import subprocess

def run_training():
    """Exécuter l'entraînement"""
    print("=" * 60)
    print("Démarrage de l'entraînement")
    print("=" * 60)
    
    script_path = os.path.join("scripts", "train.py")
    if not os.path.exists(script_path):
        print(f"Erreur: {script_path} non trouvé")
        return False
    
    # Exécuter le script d'entraînement
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    return result.returncode == 0

def run_evaluation():
    """Exécuter l'évaluation"""
    print("=" * 60)
    print("Démarrage de l'évaluation")
    print("=" * 60)
    
    script_path = os.path.join("scripts", "evaluate.py")
    if not os.path.exists(script_path):
        print(f"Erreur: {script_path} non trouvé")
        return False
    
    # Exécuter le script d'évaluation
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML pour classification de maladies rétiniennes")
    parser.add_argument("--train", action="store_true", help="Exécuter l'entraînement")
    parser.add_argument("--eval", action="store_true", help="Exécuter l'évaluation")
    parser.add_argument("--full", action="store_true", help="Exécuter l'entraînement puis l'évaluation")
    
    args = parser.parse_args()
    
    if not any([args.train, args.eval, args.full]):
        parser.print_help()
        return
    
    if args.train or args.full:
        if not run_training():
            print("Erreur pendant l'entraînement")
            return
    
    if args.eval or args.full:
        if not run_evaluation():
            print("Erreur pendant l'évaluation")
            return
    
    print("\n" + "=" * 60)
    print("Exécution terminée avec succès!")
    print("=" * 60)

if __name__ == "__main__":
    main()