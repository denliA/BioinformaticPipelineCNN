# BioinformaticPipelineCNN

Genetic pipeline inspired by Google's DeepVariant in order to dectect cancer-causing variants.


Snakemake workflow:
1)Simulate genome with VARSIM
2)Manage bioinformatics files with SAMTOOLS
3)Extract pileup images with DeepVariant
4)Train the keras CNN to detect smalls variants (insertion, deletion, snp)
5)Test the pipeline

Contenairisation with Docker and Singularity

See the bellow documents :

[manuel_d_installation_1.3.docx](https://github.com/denliA/BioinformaticPipelineCNN/files/8825005/manuel_d_installation_1.3.docx)

[manuel_d_utilisation_1.3.docx](https://github.com/denliA/BioinformaticPipelineCNN/files/8825009/manuel_d_utilisation_1.3.docx)
