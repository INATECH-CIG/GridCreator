'''
Zensus Abfrage
'''
#%%
#import pandas as pd
#import numpy as np
import pystatis
from pystatis import Table
from pystatis.helloworld import logincheck

#%%
# only run to set credentials
#pystatis.set_credentials()
# dann im fenster für genesis, zensus und regio benutzername und pw angeben
# API token funktioniert NICHT!
# dann checken ob es geht
#%%
logincheck("zensus")
#%%
t = Table(name="1000A-0000")
t.get_data()
t.name
#%%

t.data.head()
# Bevölkerungszahlen