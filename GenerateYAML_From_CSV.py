# Utility tool to generate inputs for Boltz-2 Structure and docking prediction

# The inputs are a .csv file with the SMILES strings of the molecules and the output should be .yaml file with the following format:

"""
version: 1  # Optional, defaults to 1
sequences:
  - protein:
      id: [A]
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
      msa: ./msa/CTR.a3m
  - ligand:
      id: [B]
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'

"""



import pandas as pd

# Sequence for CTR protein
sequence = "GPAAFSNQTYPTIEPKPFLYVVGRKKMMDAQYKCYDRMQQLPAYQGEGPYCNRTWDGWLCWDDTPAGVLSYQFCPDYFPDFDPSEKVTKYCDEKGVWFKHPENNRTWSNYTMCNAFTPEKLKNAYVLYYLAIVGHSLSIFTLVISLGIFVFFRSLGCQRVTLHKNMFLTYILNSMIIIIHLVEVVPNGELVRRDPVSCKILHFFHQYMMACNYFWMLCEGIYLHTLIVVAVFTEKQRLRWYYLLGWGFPLVPTTIHAITRAVYFNDNCWLSVETHLLYIIHGPVMAALVVNFFFLLNIVRVLVTKMRETHEAESHMYLKAVKATMILVPLLGIQFVVFPWRPSNKMLGKIYDYVMHSLIHFQGFFVATIYCFCNNEVQTTVKRQWAQFKIQWNQRWGRRPSNRSARAAAAAAEAGDIPIYICHQELRNEPANNQGEESAEIIPLNIIEQESSAPAGLEVLFQ"

# Read the CSV file containing the SMILE

def generate_yaml_from_csv(csv_file, output_yaml):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Prepare the YAML content
    yaml_content = "version: 1\nsequences:\n"

    # Add protein sequence
    yaml_content += "  - protein:\n"
    yaml_content += f"      id: [A]\n"
    yaml_content += f"      sequence: {sequence}\n"
    yaml_content += "      msa: ./msa/CTR.a3m\n"

    # Add ligands from the CSV file
    for index, row in df.iterrows():
        smiles = row['SMILES']
        yaml_content += "  - ligand:\n"
        yaml_content += f"      id: [{chr(66 + index)}]\n"  # B, C, D, etc.
        yaml_content += f"      smiles: '{smiles}'\n"

    # Write to output YAML file
    with open(output_yaml, 'w') as file:
        file.write(yaml_content)
    




