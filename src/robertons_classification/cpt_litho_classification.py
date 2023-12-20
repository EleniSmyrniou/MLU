from BroReader import read_BRO
import pickle
from datetime import date

if __name__ == "__main__":
    location = [117769, 439304]
    radius_distance = 1  # in km
    start_date = date(2015, 1, 1)
    c = read_BRO.read_cpts(location, radius_distance, output_dir="cpts", interpret_cpt=True)
    # open a file, where you ant to store the data
    file = open('cpts', 'wb')
    # dump information to that file
    pickle.dump(c, file)
    file.close()