from medallion.bronze import bronze
from medallion.silver import silver

def main():
    print("Iniciando o processo de ETL...")
    print("Camada Bronze: Extração de dados")
    bronze()
    print("Camada Prata: Transformação de dados")
    silver()    

if __name__ == "__main__":
    bronze()