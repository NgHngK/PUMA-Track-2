import argparse,os
p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();os.environ["PUMA_RUN_OUTPUT"]=a.output
from clean_mask import main
main()
