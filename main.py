from medallion.bronze import bronze
from medallion.silver import silver
from medallion.gold import gold

def main():
    print("Iniciando o processo de ETL...")
    print("Camada Bronze: Extração de dados")
    bronze()
    print("Camada Prata: Transformação de dados")
    silver()
    print("Camada Ouro: Carregamento dos dados")
    gold()

if __name__ == "__main__":
    main()