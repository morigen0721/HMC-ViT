# Readme
* train_code
    * train.py: Training process file
    * train_data_gen: Training data generation script
    * model: MC-ViT model
    * toolbox: Toolbox
    * weight.pth: Trained MC-ViT weights
    * environment.yml: Conda environment YML file
* fig2：gen fig2 in paper
    * val.py: Calls methods, reads data, and performs validation
    * val_data_gen.py: Generates validation data corresponding to Figure 3
    * tricnndualtran.py: CNN + Efficient DCT-ViT
    *	 partricnntran.py: MC-ViT
    * dualtran.py: DCT-ViT
    * cnndualtran.py: CNN + DCT-ViT
    * plot.py: Code for plotting
    * toolbox.py: Toolbox
    * compare.py: Encapsulation of comparison methods in this file
* fig3&4：gen fig3 in paper
    * val.py: Calls methods, reads data, and performs validation
    * val_data_gen.py: Generates validation data corresponding to Figure 4.
    * toolbox.py: Toolbox
    * prop.py: MC-ViT
    * plot.py: Code for plotting
    * dualtran.py: DCT-ViT
    * cnnlow.py: CNN
    * compare.py: Encapsulation of comparison methods in this file