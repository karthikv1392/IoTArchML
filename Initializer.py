_Author_ = "********"

# Initalizing all basic configurations

from configparser import ConfigParser
import traceback

CONFIG_FILE = "settings.conf"
CONFIG_SECTION = "settings"
CONFIG_SECTION_MODEL = "model" # For reading configurations specific to the model
CONFIG_SECTION_CC = "cupcarbon" # For reading configurations specific to cupCarbon
CONFIG_SECTION_SERVICE = "service"
CONFIG_SECTION_ADAPTATION = "adaptation"

class Initialize():
    def __init__(self):
        data_path = ""
        energy_val = 19160.0  # starting energy value as per CupCarbon
        component_count = 0  # The total number of sensors for which the monitoring needs to be done
        data_file = ""
        json_path = ""
        port = 8067 # Default port
        try:
            parser = ConfigParser()
            parser.read(CONFIG_FILE)
            self.data_path = parser.get(CONFIG_SECTION, "data_path")
            self.data_file = parser.get(CONFIG_SECTION, "data_file")
            self.json_path = parser.get(CONFIG_SECTION, "json_path")  # For storing the output time series object
            self.model_path = parser.get(CONFIG_SECTION, "model_path")  # For storing the output time series object
            self.energy_val = float(parser.get(CONFIG_SECTION, "initial_energy"))  # For getting maximum energy
            self.component_count = int(parser.get(CONFIG_SECTION, "component_count"))
            self.scalar_path = parser.get(CONFIG_SECTION, "scalar_path")  # For storing the output time series object

            # For CupCarbon data


            # For Model level
            self.epochs = int(parser.get(CONFIG_SECTION_MODEL, "epochs"))
            self.batch_size = int(parser.get(CONFIG_SECTION_MODEL, "batch_size"))
            self.num_features = int(parser.get(CONFIG_SECTION_MODEL, "num_features"))
            self.propotion_value = float(parser.get(CONFIG_SECTION_MODEL, "propotion_value"))



            # Data traffic related
            self.data_traffic_path = parser.get(CONFIG_SECTION_CC,"log_path") ## Same can be used for sensor data as well
            self.data_traffic_file = parser.get(CONFIG_SECTION_CC,"log_file")
            self.energy_path = parser.get(CONFIG_SECTION_CC,"energy_path")
            self.energy_file = parser.get(CONFIG_SECTION_CC, "energy_file")

            # Adaptation related
            self.adaptation_type = parser.get(CONFIG_SECTION_ADAPTATION,"adaptation_type")
            self.adaptation_model_json = parser.get(CONFIG_SECTION_ADAPTATION,"adaptation_model_json")
            self.adaptation_model_h5 = parser.get(CONFIG_SECTION_ADAPTATION,"adaptation_model_h5")
            self.adaptation_model_scalar = parser.get(CONFIG_SECTION_ADAPTATION,"adaptation_model_scalar")
            self.energy_hp = float(parser.get(CONFIG_SECTION_ADAPTATION,"energy_hp"))
            self.energy_bp = float(parser.get(CONFIG_SECTION_ADAPTATION,"energy_bp"))
            self.horizon = int(parser.get(CONFIG_SECTION_ADAPTATION,"horizon"))
            self.lag = int(parser.get(CONFIG_SECTION_ADAPTATION,"lag"))
            self.bp_count = int(parser.get(CONFIG_SECTION_ADAPTATION,"bp_count"))
            self.decision_period = int(parser.get(CONFIG_SECTION_ADAPTATION,"decision_period"))





            #Service Related configurations
            self.port = int(parser.get(CONFIG_SECTION_SERVICE,"port"))
            self.adaptation_file = parser.get(CONFIG_SECTION_SERVICE,"adaptation_path")
            self.request_url = parser.get(CONFIG_SECTION_SERVICE,"request_url")


        except Exception as e:
            traceback.print_exc()

