# BioinformaticPipelineCNN

Genetic pipeline inspired by Google's DeepVariant in order to detect cancer-causing alleles.

Example of a read:
![Capture d’écran 2022-06-02 à 17 05 58](https://user-images.githubusercontent.com/91119589/171660678-4fd937b2-7996-4969-8fb8-ca25d267c7e7.png)

<ol>Snakemake workflow:
  <li>Simulate genome with VARSIM</li>
  <li>Manage bioinformatics files with SAMTOOLS</li>
  <li>Extract pileup images with DeepVariant</li>
  <li>Train the keras CNN to detect smalls variants (insertion, deletion, snp)</li>
  <li>Test the pipeline</li>
</ol>
Contenairisation with Docker and Singularity

See the bellow documents :

[manuel_d_installation_1.3.docx](https://github.com/denliA/BioinformaticPipelineCNN/files/8825005/manuel_d_installation_1.3.docx)

[manuel_d_utilisation_1.3.docx](https://github.com/denliA/BioinformaticPipelineCNN/files/8825009/manuel_d_utilisation_1.3.docx)
