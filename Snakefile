fichiers_fasta = ["chr20"]
fichiers_tfrecord_gz = ["make_examples.tfrecord-00000-of-00004"]

def getLANES():
    FICHIERS_LANES = glob_wildcards(y+"/simulated.lane{nb_lane}.read.sam")
    maliste = []
    for j in range(len(list(FICHIERS_LANES))):
        for k in range(len(list(FICHIERS_LANES)[j])):
            maliste.append("simulated.lane"+str(list(FICHIERS_LANES)[j][k])+".read")
    return maliste

def getFASTA():
    maliste = []
    a = glob_wildcards(y+"/{fa}.fa")
    for w in range(len(list(a))):
        for z in range(len(list(a[w]))):
            maliste.append(a[w][z])
    return maliste

def getRECORDS():
    maliste = []
    c = glob_wildcards(y+"/dv_sur_simu/intermediate_results_dir/{tf}.gz")
    for w in range(len(list(c))):
        for z in range(len(list(c[w]))):
            s = c[w][z]
            
            if("make_examples" in s and "00000" in s):
                maliste.append(c[w][z])
    return maliste

v = config["lanes"]
y = str(config["out"])

rule all:
    input:
        expand(y+"/simulated.lane{num}.read.sam",num=[x for x in range(0,v)]),
        expand(y+"/simulated.lane{num}.read.sorted.bam.bai",num=[x for x in range(0,v)]),
        expand(y+"/simulated.lane{num}.read.sorted.bam",num=[x for x in range(0,v)]),
        expand(y+"/simu.fa"),
        expand(y+"/simu.fa.fai"),
        expand(y+"/dv_sur_simu/output.vcf.gz"),
        expand(y+"/dv_sur_simu/intermediate_results_dir/make_examples.tfrecord-00000-of-00004.gz")

rule generation_variant_varsim:
    input:
        expand("work/refs/{fasta_}.fa",fasta_=fichiers_fasta)
    output:
        out1 = expand(y+"/simulated.lane{num}.read.sam",num=[x for x in range(0,v)]),
        out2 = expand(y+"/simu.fa")

    shell:
        "/Neat/varsim/varsim.py --id simu "
        "--reference {input} "   
        "--simulator_executable /Neat/art_src_MountRainier_Linux/art_illumina " 
        "--total_coverage {config[cover]} "
        "--vc_in_vcf {config[vcf]} "
        "--sv_insert_seq work/refs/insert_seq.txt "
        "--out_dir {config[out]} --log_dir {config[log]} "
        "--work_dir {config[work]} --nlanes {config[lanes]} "
        "--read_length 200 "
        "--disable_rand_dgv "
        "--vc_num_ins {config[ins]} "
        "--vc_num_snp {config[snp]} "
        "--vc_num_del {config[del]} "
        "--art_options '\-sam' {config[add]}"
        

rule sam_to_bam:
    input:
        "{yes}.sam"
    output:
        "{yes}.bam"
    shell:
        "samtools view -S -b {input} > {output}"

rule sort_bam:
    input:
        "{yes}.bam"
    output:
        "{yes}.sorted.bam"
    shell:
        "samtools sort {input} -o {output}"

rule bam_to_bam_bai:
    input:
        "{yes}.sorted.bam"

    output:
        "{yes}.sorted.bam.bai"

    shell:
        "samtools index {input}"

rule fasta_to_fasta_fai:
    input:
        "{no}.fa"
    output:
        "{no}.fa.fai"
    shell:
        "samtools faidx {input}"


fichiers_sam_list = getLANES()
simulated_fasta = getFASTA()
fichiers_tfrecord_gz = getRECORDS()


rule lance_deepvariant:
    input:
        a = expand(y+"/simulated.lane{num}.read.sorted.bam",num=[x for x in range(0,v)]),
        b = expand(y+"/simu.fa"),
        c = expand(y+"/simu.fa.fai"),
        d = expand(y+"/simulated.lane{num}.read.sorted.bam.bai",num=[x for x in range(0,v)]),

    shell:        
        "singularity run -B /usr/lib/locale/:/usr/lib/locale/ docker://google/deepvariant:\"1.3.0\" /opt/deepvariant/bin/run_deepvariant   "
        "--model_type=WGS   "
        "--ref={input.b} "  
        "--reads={input.a}  "
        "--regions  \"{config[regions]}\"   "
        "--output_vcf={config[out]}/dv_sur_simu/output.vcf.gz   "
        "--output_gvcf={config[out]}/dv_sur_simu/output.g.vcf.gz "
        "--intermediate_results_dir {config[out]}/dv_sur_simu/intermediate_results_dir   "
        "--num_shards={config[coeurs]} "
    
rule see_vcf:
    input:
        y+"/dv_sur_simu/output.vcf.gz"
    shell:
        "sudo gunzip -k {input}"

rule extract_img:
    input:
        expand(y+"/dv_sur_simu/intermediate_results_dir/{tf_gz}.gz",tf_gz=fichiers_tfrecord_gz)
    output:
        y+"/dv_images/images"
    shell:
        "python img_extract.py {input} {config[out]}"


rule train:
    input:
        a = y+"/dv_images/images"
        
    output:
        y+"/dv_imagesDivided/test/images"
        
    shell:
        "python CNN/trainCNN.py {input.a} {config[batch]} {config[epoch]}"

rule test:
    input:
        y+"/dv_imagesDivided/test/images"
    shell:
        "python CNN/testCNN.py {input} {config[batch]} {config[rvcf]}"