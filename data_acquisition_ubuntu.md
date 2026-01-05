# GenoPHI 数据集获取指南 (Ubuntu环境)

## 1. 使用项目内置数据集

GenoPHI项目本身已包含多个示例数据集，位于 `data/` 目录下：

```bash
# 查看项目内置数据集
ls -la /path/to/GenoPHI/data/
```

### 交互矩阵文件（位于 `data/interation_matrices/`）
- `ecoli_interaction_matrix.csv`：大肠杆菌噬菌体-宿主交互数据
- `klebsiella1_interaction_matrix.csv`：克雷伯菌交互数据
- `pseudomonas_interaction_matrix.csv`：假单胞菌交互数据
- `vibrio_interaction_matrix.csv`：弧菌交互数据

### 实验验证数据（位于 `data/experimental_validation/`）
- `BASEL_ECOR_interaction_matrix.csv`：ECOR菌株与Bas噬菌体交互数据
- `ECOR27_TnSeq_high_fitness_genes.csv`：TnSeq实验数据

## 2. 从公共数据库下载数据

### 2.1 使用 NCBI Entrez Direct 工具

首先安装 Entrez Direct：

```bash
sudo apt-get update
sudo apt-get install -y ncbi-entrez-direct
```

#### 下载蛋白质序列（FASTA格式）

```bash
# 下载特定GenBank ID的蛋白质序列
efetch -db protein -id "NP_040550" -format fasta > protein.fasta

# 批量下载多个蛋白质序列
efetch -db protein -id "NP_040550,NP_040551,NP_040552" -format fasta > proteins.fasta
```

#### 下载基因组序列

```bash
# 下载特定基因组序列
efetch -db nucleotide -id "NC_000913" -format fasta > genome.fasta
```

### 2.2 使用 UniProt API

```bash
# 下载特定UniProt ID的蛋白质序列
curl -o protein.fasta "https://rest.uniprot.org/uniprotkb/P00502.fasta"

# 批量下载多个UniProt ID的蛋白质序列
# 创建包含UniProt ID的文件（每行一个ID）
echo -e "P00502\nP00503\nP00504" > uniprot_ids.txt

# 批量下载
while read id; do
    curl -o "${id}.fasta" "https://rest.uniprot.org/uniprotkb/${id}.fasta"
done < uniprot_ids.txt

# 合并为单个文件
cat *.fasta > combined_proteins.fasta
rm P00502.fasta P00503.fasta P00504.fasta
```

### 2.3 使用 SRA Toolkit 下载测序数据

```bash
# 安装 SRA Toolkit
sudo apt-get install -y sra-toolkit

# 下载SRA数据并转换为FASTQ格式
prefetch SRR1234567
fastq-dump --split-files SRR1234567
```

## 3. 生成符合GenoPHI格式的数据

### 3.1 从基因组序列生成蛋白质序列（使用 Prokka）

```bash
# 安装 Prokka
sudo apt-get install -y prokka

# 使用 Prokka 预测蛋白质序列
prokka --outdir prokka_output --prefix sample genome.fasta

# 生成的蛋白质序列文件位于：prokka_output/sample.faa
```

### 3.2 准备交互矩阵文件

GenoPHI需要包含以下列的CSV文件：
- `strain`：宿主菌株名称
- `phage`：噬菌体名称
- `interaction`：交互结果（1表示感染，0表示不感染）

示例交互矩阵文件：
```csv
strain,phage,interaction
Ecoli_001,Phage_001,1
Ecoli_001,Phage_002,0
Ecoli_002,Phage_001,1
Ecoli_002,Phage_002,1
```

## 4. 数据格式验证

确保下载或生成的数据符合GenoPHI要求：

```bash
# 检查FASTA文件格式
head -n 10 protein.fasta

# 检查CSV交互矩阵格式
head -n 5 interaction_matrix.csv
```

## 5. 用于GenoPHI分析的示例命令

```bash
# 使用项目内置的大肠杆菌数据运行蛋白质家族工作流
python -m genophi protein-family-workflow \
    --input_strain data/strain_proteins/ecoli_proteins.faa \
    --phenotype_matrix data/interation_matrices/ecoli_interaction_matrix.csv \
    --output results/ecoli_analysis

# 使用k-mer工作流
python -m genophi kmer-workflow \
    --input_strain data/strain_proteins/ecoli_proteins.faa \
    --phenotype_matrix data/interation_matrices/ecoli_interaction_matrix.csv \
    --output results/ecoli_kmer_analysis
```

## 6. 公共数据库资源

- [NCBI Protein Database](https://www.ncbi.nlm.nih.gov/protein/)
- [UniProt Knowledgebase](https://www.uniprot.org/)
- [PDB Protein Database](https://www.rcsb.org/)
- [IMG/VR - Viral Database](https://img.jgi.doe.gov/cgi-bin/vr/main.cgi)
- [PhagesDB](https://phagesdb.org/)

## 7. 注意事项

1. 确保下载的数据与GenoPHI的输入格式兼容
2. 大型数据集可能需要较长下载时间和较大存储空间
3. 某些数据库可能需要注册或API密钥（如批量下载）
4. 请遵守各数据库的使用条款和引用要求