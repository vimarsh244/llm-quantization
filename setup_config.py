"""
Helper script to set up configuration from examples.
"""

import json
import sys

def list_configs():
    """List available example configurations."""
    with open('config_examples.json', 'r') as f:
        examples = json.load(f)
    
    print("\nAvailable configurations:")
    print("-" * 80)
    for name, config in examples['configurations'].items():
        description = config.get('description', 'No description')
        print(f"  {name:<25} - {description}")
    print("-" * 80)
    print()


def setup_config(config_name: str):
    """Set up config.json from an example."""
    with open('config_examples.json', 'r') as f:
        examples = json.load(f)
    
    if config_name not in examples['configurations']:
        print(f"ERROR: Configuration '{config_name}' not found")
        list_configs()
        sys.exit(1)
    
    config = examples['configurations'][config_name]
    
    # Remove description field before saving
    config_to_save = {k: v for k, v in config.items() if k != 'description'}
    
    with open('config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"✓ Configuration '{config_name}' loaded into config.json")
    print(f"  Model: {config['model_name']}")
    print(f"  Methods: {', '.join(config['quantization_methods'])}")
    print(f"  Description: {config.get('description', 'N/A')}")
    print()
    print("To run benchmark:")
    print("  python benchmark_runner.py config.json")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_config.py [config_name]")
        list_configs()
        sys.exit(0)
    
    config_name = sys.argv[1]
    
    if config_name == "list":
        list_configs()
    else:
        setup_config(config_name)


if __name__ == "__main__":
    main()

