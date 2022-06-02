# BioinformaticPipelineCNN

Genetic pipeline inspired by Google's DeepVariant in order to detect cancer-causing alleles.


Snakemake workflow:
<ol>Simulate genome with VARSIM
  <li>Manage bioinformatics files with SAMTOOLS</li>
  <li>Extract pileup images with DeepVariant</li>
  <li>Train the keras CNN to detect smalls variants (insertion, deletion, snp)</li>
  <li>Test the pipeline</li>
</ol>
Contenairisation with Docker and Singularity

See the bellow documents :

[manuel_d_installation_1.3.docx](https://github.com/denliA/BioinformaticPipelineCNN/files/8825005/manuel_d_installation_1.3.docx)

[manuel_d_utilisation_1.3.docx](https://github.com/denliA/BioinformaticPipelineCNN/files/8825009/manuel_d_utilisation_1.3.docx)
