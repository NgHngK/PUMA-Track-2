import argparse,os
p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--weights",required=True);a=p.parse_args();os.environ["PUMA_RUN_OUTPUT"]=a.output;os.environ["PUMA_UNI2_WEIGHTS"]=a.weights
from restricted_lora import main
main()
