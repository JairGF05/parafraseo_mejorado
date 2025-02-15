import pandas as pd
import os
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Desactivar el uso de MPS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Cargar modelo de parafraseo con BART
paraphrase_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def parafrasear_texto(texto):
    try:
        if not texto or not isinstance(texto, str):
            return ""
        resultado = paraphrase_pipeline(texto, max_length=200, min_length=30, do_sample=False)
        return resultado[0]['generated_text']
    except Exception as e:
        print(f"Error al parafrasear: {e}")
        return texto

def procesar_fila(row):
    try:
        if row.isnull().all():
            return None
        if "text" in row:
            texto_original = str(row["text"])
            texto_parafraseado = parafrasear_texto(texto_original)
            return texto_parafraseado
    except Exception as e:
        print(f"Error al procesar la fila: {e}")
    return None

def extraer_y_parafrasear(input_excel, output_file, chunksize=20):
    try:
        hojas = pd.ExcelFile(input_excel)
    except Exception as e:
        print(f"Error al leer el archivo Excel: {e}")
        return
    
    with open(output_file, 'w', encoding='utf-8') as archivo_salida:
        for nombre_hoja in hojas.sheet_names:
            print(f"Procesando hoja: {nombre_hoja}")
            
            try:
                df = pd.read_excel(input_excel, sheet_name=nombre_hoja)
                total_rows = len(df)
                
                for start in tqdm(range(0, total_rows, chunksize), desc=f"Procesando {nombre_hoja}", unit="chunk"):
                    chunk = df.iloc[start:start + chunksize]
                    
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        resultados = list(executor.map(procesar_fila, [row for _, row in chunk.iterrows()]))
                    
                    for resultado in resultados:
                        if resultado:
                            archivo_salida.write(f"{resultado}\n")
            except Exception as e:
                print(f"Error al procesar la hoja {nombre_hoja}: {e}")
                continue
    
    print(f"Texto parafraseado guardado en: {output_file}")

def main():
    input_excel = "corpus.xlsx"
    output_file = "parafraseo.txt"
    extraer_y_parafrasear(input_excel, output_file)

if __name__ == "__main__":
    main()