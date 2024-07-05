import sys
from utils import load_json_config, AppDataGenerator


if __name__ == '__main__':
    app_data_generator = AppDataGenerator(load_json_config(sys.argv[1]))
    app_tag = sys.argv[2]
    app_data_generator.generate_app_data(app_tag)
