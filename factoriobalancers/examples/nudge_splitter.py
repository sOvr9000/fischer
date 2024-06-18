
import numpy as np
from fischer.factoriobalancers import BeltGrid
from fischer.factoriobps import load_blueprint_string, get_blueprint_string



def main():
    # Initialize a large grid
    grid = BeltGrid((9, 9), max_inputs=4, max_outputs=7, max_splitters=11, max_turtles=1)

    # Load a 4-7 balancer (dimensions of which are 9x9) into the grid
    grid.load_bp_data(load_blueprint_string('0eNqlWNtu4yAQ/ZWIZ7syF98idX+kiqqkRRWSgy0bV40i//uSuk27a4+BIS9JsDlnODAHhis5NaPseqUN2V+Jemn1QPZPVzKoN31sbm3m0kmyJ++qN6NtSYg+nm8N8xupIFNClH6VH2RPpySgZ/mrJ5sOCZHaKKPkHMDnn8uzHs8n2Vvoe2/50fVyGFLTH/XQtb1JT7IxFr1rB9u91TdqC5nyhzwhF/uDPeSW6VX18mV+Lm5x/kfAwgkYSFCsEPBwAvqbYAVSREDydcg8HDJzQBbBkC7EcoE4dI0yxj5bYm0qWAXH5hKwDkZkDkSaBQyXbQ6XhqcRd0UXnjjCBckDBiy2BywifCPz8Q2aRzBQL4YiwpoyQJeQFErZT7RrWFUIFt2Oq8abxWK62JrNL5NptFtQ/9a39ttFMc/X177WjqYbDVkjoREuCqjMGN5YALEZx7ufn9gCITYNFTsPWH58W+MC72UQZImHhKatwjsO9Zq2Gk/g5Zk8wx9cvEbAKd4y/QgijovAtHKOSJa7LAvdC4/U4QLvU8B653mEv/6T8UpDURfBxz9I8hIRLHMrDodeRRiipzo1+oiX+ZQxIgs+nAFLRVC0NQLzKVhMCvFF4rvVFhzvA9zHaITAE3jVvSLHe7FX3SsKvMn4EYScZekddw2pCnYWoIQR+DKQ+SyLPLwmhAp+fEkIqJjjK0IIkaMRoVFjTqbcbRTwXptH3KwIYBD4mxUIsURXBBBihU6ET8RDQpSRZ9v956IyIc3RdrVtT193i+l8pfj4fSd52O1S+/mzA14oDxbkXfbDPH0VFWXNyqqoM56JafoLE/gFRw=='))
    print(grid)

    # Nudge a splitter and reconnect its inbound and outbound belts.
    grid.nudge_splitter((4, 4), (4, 5))

    bp_str = get_blueprint_string(grid.to_blueprint_data())
    print(bp_str)



if __name__ == '__main__':
    main()


