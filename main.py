from medallion.bronze import bronze
from medallion.silver import silver

def main():
    try:
        print("Iniciando o processo de ETL...")

        print("Camada Bronze: Extração de dados")
        bronze()

        print("Camada Prata: Transformação de dados")
        silver()

    except Exception as e:
        print(f"Pipeline interrompido: {e}")


if __name__ == "__main__":
    main()