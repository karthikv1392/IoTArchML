_Author_ = "Karthik Vaidhyanathan"
import random
import time

# Program to generate the frequency values for simulation data phase


class Random_Frequency_Generator():
    def __init__(self):
        self.dict_sensorkeys  = {"v1EnNorm": 20000,"v1EnCrit":5000,"v1ExNorm":20000,"v1ExCrit":5000,"v2EnNorm":20000,"v2EnCrit":5000,"v2ExNorm":20000,"v2ExCrit":5000,"v3EnNorm":20000,
                            "v3EnCrit":5000,"v3ExNorm":20000,"v3ExCrit":5000,"p1EnNorm":60000,"p1EnCrit":10000,"p1ExNorm":30000,"p1ExCrit":10000,"p2EnNorm":60000,"p2EnCrit":10000,"p2ExNorm":30000,"p2ExCrit":10000}

    def generate_frequency(self):

        while(1):
            write_string = ""

            for key in self.dict_sensorkeys:
                increment = 0
                if "Crit" in key:
                    increment = (random.randint(0,5)*1000)
                    print (increment)
                    update_val = self.dict_sensorkeys[key] + increment
                    write_string = write_string + key + " " + str(update_val) + "\n"
                else:
                    increment = (random.randint(0, 20) * 1000)
                    update_val = self.dict_sensorkeys[key] + increment
                    write_string = write_string + key + " " + str(update_val) + "\n"

            print (self.dict_sensorkeys)
            write_string = write_string[:-1]
            text_file = open("config.txt", "r+")
            text_file.write(write_string)
            text_file.close()
            time.sleep(25)







if __name__ == '__main__':
    random_frequency_generator_obj =  Random_Frequency_Generator()
    random_frequency_generator_obj.generate_frequency()