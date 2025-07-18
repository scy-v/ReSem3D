import os
import argparse
from environment import OGEnv
from arguments import get_config
from interfaces import setup_LMP
from visualizer import Visualizer

def run(config, scene_path, objects_path, instruction):
    """Set up the environment and run ReSem3D."""
    # Set up environment
    env = OGEnv(config['env'], scene_path, objects_path, verbose=False)
    env.vision_inference()

    # Set up visualizer and LMP modules
    visualizer = Visualizer(config, env)
    lmps = setup_LMP(env, config['env'], visualizer, debug=False)

    # Execute the instruction through the ReSem3D task planner UI
    ReSem3D_UI = lmps["task_planner_ui"]
    ReSem3D_UI(instruction)


def main():
    """Program entry point: parse args, load config, and start running."""
    parser = argparse.ArgumentParser(description="Run ReSem3D with configurable parameters.")
    parser.add_argument("--load_cache", action="store_true", help="Enable cache in all LMP configs.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualizer display.")
    parser.add_argument("--instruction", type=str, default="Please pick up the erlenmeyer flask and place it on the magnetic stirrer.",
                        help="Natural language instruction for ReSem3D.")
    args = parser.parse_args()

    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "configs", "omnigibson_config")
    config_file = os.path.join(config_dir, "config.yaml")
    scene_path = os.path.join(config_dir, "og_scene_file.json")
    objects_path = os.path.join(config_dir, "objects.yaml")

    # Load config
    config = get_config(config_path=config_file)

    # Add visualize
    config['env']['visualize'] = args.visualize
    
    # Add load_cache into each LMP config
    for _, lmp_cfg in config['env']['lmp_config']['lmps'].items():
        if isinstance(lmp_cfg, dict):
            lmp_cfg['load_cache'] = args.load_cache

    # Start running
    run(config, scene_path, objects_path, args.instruction)


if __name__ == "__main__":
    main()
