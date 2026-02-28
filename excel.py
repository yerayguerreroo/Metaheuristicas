from openpyxl import Workbook
import json
import os

def exportar_reporte_excel(resumen, dir, filename="reporte_SA.xlsx"):
    """
    Exporta un diccionario tipo resumen a un archivo Excel.
    Mantiene tu formato: exportar_reporte_excel(resumen, dir, filename).
    """

    def to_excel_value(x):
        """Convierte tipos no soportados (listas/dicts/tuplas/sets) a texto."""
        if x is None:
            return ""
        if isinstance(x, (list, tuple, dict, set)):
            return json.dumps(x, ensure_ascii=False)
        return x

    wb = Workbook()

    # =========================
    # HOJA 1 — RESUMEN GLOBAL
    # =========================
    ws_resumen = wb.active
    ws_resumen.title = "Resumen"

    ws_resumen.append(["Métrica", "Valor"])

    for key in resumen:
        if key != "resultados_individuales":
            ws_resumen.append([key, to_excel_value(resumen[key])])

    # =========================
    # HOJA 2 — EJECUCIONES
    # =========================
    ws_runs = wb.create_sheet("Ejecuciones")

    resultados = resumen.get("resultados_individuales", [])

    if len(resultados) > 0:
        # Encabezados (usar list para poder iterar varias veces)
        headers = list(resultados[0].keys())
        ws_runs.append(headers)

        # Filas
        for r in resultados:
            ws_runs.append([to_excel_value(r.get(h)) for h in headers])

    # Guardar archivo (asegurar dir correcto)
    os.makedirs(dir, exist_ok=True)  # crea carpeta si no existe
    path = os.path.join(dir, filename)
    wb.save(path)

    return path