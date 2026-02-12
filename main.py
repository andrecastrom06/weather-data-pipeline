from medallion.bronze import bronze
from medallion.silver import silver
from medallion.gold import gold

def main():
    try:
        print("Iniciando o processo de ETL...")

        print("Camada Bronze: Extração de dados")
        bronze()

        print("Camada Prata: Transformação de dados")
        silver()

        print("Camada Ouro: Agregação de valores analíticos")
        gold()

    except Exception as e:
        print(f"Pipeline interrompido: {e}")


if __name__ == "__main__":
    main()