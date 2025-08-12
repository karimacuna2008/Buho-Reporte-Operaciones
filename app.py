import streamlit as st
import pandas as pd
import re
import tempfile
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
from gspread.utils import rowcol_to_a1

st.set_page_config(page_title="Reporte Mensual Operaciones", layout="wide")

# ————————————————————————————————————————————————————————
# CREDENCIALES y CLIENTE de Google Sheets / Drive
# ————————————————————————————————————————————————————————
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
SPREADSHEET_ID = st.secrets["app"]["spreadsheet_id"]

creds  = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
client = gspread.authorize(creds)


# ————————————————————————————————————————————————————————
# BLOQUE 1: TRANSFORMACIÓN DE FECHAS (UTC -> America/Monterrey)
# ————————————————————————————————————————————————————————
def strip_subseconds(s: str) -> str:
    if pd.isna(s): return s
    return re.sub(r'\.\d+(?=\+)', '', s)

def transform_dates_no_subs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tz_local = ZoneInfo('America/Monterrey')
    for col in ['fecha_inicio','fecha_fin']:
        df[col] = (
            df[col].astype(str)
                   .apply(strip_subseconds)
                   .pipe(pd.to_datetime, utc=True, errors='coerce')
                   .dt.tz_convert(tz_local)
                   .dt.tz_localize(None)
        )
    return df

def fix_invalid_end_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask_diff = df['fecha_fin'].dt.date != df['fecha_inicio'].dt.date
    if mask_diff.any():
        df.loc[mask_diff, 'fecha_fin'] = df.loc[mask_diff].apply(
            lambda r: r['fecha_fin'].replace(
                year=r['fecha_inicio'].year,
                month=r['fecha_inicio'].month,
                day=r['fecha_inicio'].day
            ), axis=1
        )
    mask_early = df['fecha_fin'] < df['fecha_inicio']
    if mask_early.any():
        df.loc[mask_early, 'fecha_fin'] += pd.Timedelta(hours=12)
    return df

def load_and_transform(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = transform_dates_no_subs(df)
    return fix_invalid_end_dates(df)

# ————————————————————————————————————————————————————————
# BLOQUE 2: AGRUPAR POR MES y CÁLCULO DE HORAS EXTRA
# ————————————————————————————————————————————————————————
def group_by_month(df: pd.DataFrame) -> pd.DataFrame:
    meses = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',
             7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}
    df = df.copy()
    df['mes'] = df['fecha_inicio'].dt.month.map(meses)

    def semana_del_mes(fecha):
        primera = fecha.replace(day=1)
        wd0 = primera.weekday()  # 0=Mon ... 6=Sun
        start_sem1 = primera + timedelta(days=(7-wd0)) if wd0 >= 5 else primera
        end_sem1   = start_sem1 + timedelta(days=(6 - start_sem1.weekday()))
        if fecha <= end_sem1:
            return 1
        delta = (fecha - end_sem1).days
        return 2 + (delta - 1) // 7

    df['semana_mes'] = df['fecha_inicio'].apply(semana_del_mes)
    return df

def calculate_overtime(df: pd.DataFrame, work_start: time, work_end: time) -> pd.DataFrame:
    df = df.copy()
    def split_hours(r):
        inicio, fin = r['fecha_inicio'], r['fecha_fin']
        if pd.isna(inicio) or pd.isna(fin):
            return pd.Series({'horas_extra':0.0,'duracion':0.0})
        total_h = (fin - inicio).total_seconds()/3600
        dia = inicio.date()
        j_start = datetime.combine(dia, work_start)
        j_end   = datetime.combine(dia, work_end)
        hrs_lab = max((min(fin, j_end) - max(inicio, j_start)).total_seconds()/3600, 0)
        hrs_ext = total_h - hrs_lab
        return pd.Series({'horas_extra':round(hrs_ext,2), 'duracion':round(hrs_lab+hrs_ext,2)})
    df[['horas_extra','duracion']] = df.apply(split_hours, axis=1)
    return df

# ————————————————————————————————————————————————————————
# BLOQUE 3: CÁLCULO DE MÉTRICAS y FILAS IGNORADAS
# ————————————————————————————————————————————————————————
def compute_monthly_activity_metrics(df: pd.DataFrame):
    excepciones = ["Inventario"]
    mask_base = (df['eventuales'].fillna(0)>0) & (df['mesas'].fillna(0)>0)
    df_base   = df[mask_base]
    df_others = df[~mask_base]
    mask_zero = (df_base['resultado']==0) & (~df_base['actividad'].isin(excepciones))
    df_zero   = df_base[mask_zero]
    df_ignored = pd.concat([df_others, df_zero], ignore_index=True)

    abs_grp = df.groupby('actividad').agg(
        Frecuencia=('actividad','size'),
        total_producido=('resultado','sum'),
        tiempo_total_dec=('duracion','sum'),
        tiempo_extra_dec=('horas_extra','sum'),
        prom_tiempo_inactivo=('tiempo_inactivo','mean')
    ).reset_index().rename(columns={'actividad':'Actividad'})

    abs_grp['Total producido'] = abs_grp.apply(
        lambda r: "N/A" if r['Actividad'] in excepciones else round(r['total_producido'],2), axis=1
    )

    def dec_to_hhmmss(x):
        h=int(x); m=int((x-h)*60); s=int(round(((x-h)*60-m)*60))
        return f"{h:02d}:{m:02d}:{s:02d}"

    abs_grp['Tiempo total']       = abs_grp['tiempo_total_dec'].apply(dec_to_hhmmss)
    abs_grp['Tiempo extra total'] = abs_grp['tiempo_extra_dec'].apply(dec_to_hhmmss)
    abs_grp['Promedio tiempo inactivo'] = abs_grp['prom_tiempo_inactivo'].round(2)

    sum_ev   = df.groupby('actividad')['eventuales'].sum().fillna(0)
    mean_ev  = df[df['eventuales'].fillna(0)>0].groupby('actividad')['eventuales'].mean()
    sum_ms   = df.groupby('actividad')['mesas'].sum().fillna(0)
    mean_ms  = df[df['mesas'].fillna(0)>0].groupby('actividad')['mesas'].mean()

    prom_list = []
    for _, r in abs_grp.iterrows():
        act = r['Actividad']
        pe = round(mean_ev.get(act,0),2) if sum_ev.get(act,0)>0 else "Sin registros"
        pm = round(mean_ms.get(act,0),2) if sum_ms.get(act,0)>0 else "Sin registros"
        prom_list.append((act,pe,pm))
    crit = pd.DataFrame(prom_list, columns=['Actividad','Promedio de eventuales','Promedio de mesas'])

    metrics_df = abs_grp[[
        'Actividad','Frecuencia','Total producido',
        'Tiempo total','Tiempo extra total','Promedio tiempo inactivo'
    ]].merge(crit, on='Actividad', how='left')

    return metrics_df, df_ignored

# ————————————————————————————————————————————————————————
# BLOQUE 4: ESCRITURA EN GOOGLE SHEETS
# ————————————————————————————————————————————————————————
def write_monthly_metrics_and_ignored(metrics_df, df_ignored, month_name, year):
    abbr = month_name[:3].upper()
    yy   = str(year)[-2:]
    title = f"{abbr} {yy}"
    ss = client.open_by_key(SPREADSHEET_ID)
    try:
        ws = ss.worksheet(title)
        ss.del_worksheet(ws)
    except gspread.WorksheetNotFound:
        pass

    df2 = (df_ignored
           .drop(columns=['mes','semana_mes','tiempo_inactivo','tiempo_comida'], errors='ignore')
           .rename(columns={
               'id':'ID','fecha_inicio':'Inicio','fecha_fin':'Fin','cliente':'Cliente',
               'campana':'Campaña','tipo':'Tipo de actividad','actividad':'Actividad',
               'linea':'Linea','eventuales':'Eventuales','mesas':'Mesas',
               'nota_inicial':'Nota inicial','nota_final':'Nota final',
               'resultado':'Resultado','internos':'Internos','responsable':'Responsable',
               'horas_extra':'Horas extra','duracion':'Duracion total'
           }))

    rows_m, cols_m = metrics_df.shape
    rows_i, cols_i = df2.shape
    total_cols = max(cols_m, cols_i)

    ws = ss.add_worksheet(title=title,
                          rows=str(rows_m + rows_i + 5),
                          cols=str(total_cols + 2))

    set_with_dataframe(ws, metrics_df, row=1, col=1)
    set_with_dataframe(ws, df2,         row=rows_m+3, col=1)

    requests = []

    # Encabezado métricas
    requests.append({
      "repeatCell": {
        "range": {"sheetId": ws.id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": cols_m},
        "cell": {"userEnteredFormat": {
          "backgroundColor": {"red":1.0,"green":0.62,"blue":0.4},
          "textFormat": {"bold": True},
          "horizontalAlignment": "CENTER"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
      }
    })

    # Encabezado ignorados
    requests.append({
      "repeatCell": {
        "range": {"sheetId": ws.id, "startRowIndex": rows_m+2, "endRowIndex": rows_m+3,
                  "startColumnIndex": 0, "endColumnIndex": cols_i},
        "cell": {"userEnteredFormat": {
          "backgroundColor": {"red":1.0,"green":0.62,"blue":0.4},
          "textFormat": {"bold": True},
          "horizontalAlignment": "CENTER"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
      }
    })

    # Separador
    requests.append({
      "updateDimensionProperties": {
        "range": {"sheetId": ws.id, "dimension": "ROWS",
                  "startIndex": rows_m+1, "endIndex": rows_m+2},
        "properties": {"pixelSize": 50}, "fields": "pixelSize"
      }
    })

    # Auto-resize
    requests.append({
      "autoResizeDimensions": {
        "dimensions": {"sheetId": ws.id, "dimension": "COLUMNS",
                       "startIndex": 0, "endIndex": total_cols}
      }
    })

    # Formato condicional ceros en Eventuales/Resultado
    col_event = df2.columns.get_loc('Eventuales')
    col_res   = df2.columns.get_loc('Resultado')
    requests.append({
      "addConditionalFormatRule": {
        "rule": {
          "ranges": [{"sheetId": ws.id,
                      "startRowIndex": rows_m+3, "endRowIndex": rows_m+3+rows_i,
                      "startColumnIndex": col_event, "endColumnIndex": col_res+1}],
          "booleanRule": {"condition": {"type": "NUMBER_EQ", "values":[{"userEnteredValue":"0"}]},
                          "format": {"backgroundColor":{"red":1,"green":0.8,"blue":0.8}}}
        }, "index": 0
      }
    })

    # Formato condicional Responsable no vacío
    col_resp = df2.columns.get_loc('Responsable')
    requests.append({
      "addConditionalFormatRule": {
        "rule": {
          "ranges": [{"sheetId": ws.id,
                      "startRowIndex": rows_m+3, "endRowIndex": rows_m+3+rows_i,
                      "startColumnIndex": col_resp, "endColumnIndex": col_resp+1}],
          "booleanRule": {"condition": {"type":"NOT_BLANK"},
                          "format": {"backgroundColor":{"red":1,"green":1,"blue":0.6}}}
        }, "index": 1
      }
    })

    ss.batch_update({ "requests": requests })

def write_global_overview(params: dict, month_name: str, year: int):
    ss = client.open_by_key(SPREADSHEET_ID)
    try:
        ws = ss.worksheet("Global")
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title="Global", rows="200", cols="30")

    filas = [
        "DATOS",
        "Picks en línea",
        "Movimientos por acabados",
        "Costo total de mano de obra",
        "Costo por pick",
        "Guías",
        "$MO Eventuales",
        "$MO/Guía"
    ]
    tablas = ["FDA", "FABE", "DQ"]
    abbr = month_name[:3].upper()
    yy   = str(year)[-2:]
    header_label = f"{abbr} {yy}".upper()
    colors = {
        "FDA":  {"hdr": {"red":0.9,"green":0.3,"blue":0.3}, "dat": {"red":1.0,"green":0.8,"blue":0.8}},
        "FABE": {"hdr": {"red":0.3,"green":0.6,"blue":0.9}, "dat": {"red":0.8,"green":0.9,"blue":1.0}},
        "DQ":   {"hdr": {"red":0.3,"green":0.9,"blue":0.5}, "dat": {"red":0.8,"green":1.0,"blue":0.9}}
    }

    batch = []
    cur_row = 1

    for tabla in tablas:
        p = params[tabla]

        title_range = {
            "sheetId": ws.id,
            "startRowIndex":  cur_row-1, "endRowIndex": cur_row,
            "startColumnIndex": 0,        "endColumnIndex": 2
        }
        batch += [
            {"unmergeCells": {"range": title_range}},
            {"mergeCells":   {"range": title_range, "mergeType": "MERGE_ALL"}},
            {"repeatCell": {
                "range": title_range,
                "cell": {
                    "userEnteredValue": {"stringValue": tabla},
                    "userEnteredFormat": {
                        "backgroundColor": {"red":1.0,"green":0.62,"blue":0.2},
                        "textFormat": {"bold": True},
                        "horizontalAlignment": "CENTER",
                        "numberFormat": {"type": "TEXT"}
                    }
                },
                "fields": "userEnteredValue,userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,numberFormat)"
            }}
        ]

        hdr_row  = cur_row + 1
        data_end = hdr_row + len(filas) - 1
        rows_A = [[v] for v in filas]
        ws.update(f"A{hdr_row}:A{data_end}", rows_A)

        batch.append({
            "repeatCell": {
                "range": {"sheetId": ws.id,
                          "startRowIndex": hdr_row-1, "endRowIndex": data_end,
                          "startColumnIndex": 0, "endColumnIndex": 1},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "LEFT",
                    "numberFormat": {"type": "TEXT"},
                    "backgroundColor": {"red":1,"green":1,"blue":1}
                }},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment,numberFormat,backgroundColor)"
            }
        })

        existing = ws.row_values(hdr_row)
        new_col = len(existing) + 1

        cell = ws.cell(hdr_row, new_col)
        cell.value = header_label
        ws.update_cells([cell])

        c = colors[tabla]
        batch.append({
            "repeatCell": {
                "range": {"sheetId": ws.id,
                          "startRowIndex": hdr_row-1, "endRowIndex": hdr_row,
                          "startColumnIndex": new_col-1, "endColumnIndex": new_col},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": c["hdr"],
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "CENTER",
                    "numberFormat": {"type": "TEXT"}
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,numberFormat)"
            }
        })

        data_start = hdr_row + 1
        vals = [
            [p['picks_linea']],
            [p['movimientos_acabados']],
            [p['costo_mo']],
            [round(p['costo_mo']/p['picks_linea'],2) if p['picks_linea'] else 0],
            [p['num_guias']],
            [p['mo_eventuales']],
            [round(p['mo_eventuales']/p['num_guias'],2) if p['num_guias'] else 0]
        ]
        rng = f"{rowcol_to_a1(data_start,new_col)}:{rowcol_to_a1(data_end,new_col)}"
        ws.update(rng, vals)

        batch.append({
            "repeatCell": {
                "range": {"sheetId": ws.id,
                          "startRowIndex": data_start-1, "endRowIndex": data_end,
                          "startColumnIndex": new_col-1, "endColumnIndex": new_col},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": c["dat"],
                    "horizontalAlignment": "LEFT",
                    "numberFormat": {"type": "TEXT"}
                }},
                "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,numberFormat)"
            }
        })

        batch.append({
            "updateDimensionProperties": {
                "range": {"sheetId": ws.id, "dimension": "ROWS",
                          "startIndex": data_end, "endIndex": data_end+1},
                "properties": {"pixelSize": 50},
                "fields": "pixelSize"
            }
        })

        cur_row = data_end + 2

    if batch:
        ss.batch_update({ "requests": batch })

# ————————————————————————————————————————————————————————
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ————————————————————————————————————————————————————————
def run_monthly_pipeline(path: str, mes: str, year: int,
                         update_report: bool, global_params: dict,
                         work_start: time, work_end: time):
    df = load_and_transform(path)
    df = group_by_month(df)
    df = df[df['mes']==mes]
    df = calculate_overtime(df, work_start, work_end)

    metrics_df, df_ignored = compute_monthly_activity_metrics(df)
    if update_report:
        write_monthly_metrics_and_ignored(metrics_df, df_ignored, mes, year)
        write_global_overview(global_params, mes, year)

    return {'metrics': metrics_df, 'ignored_rows': df_ignored}

# ————————————————————————————————————————————————————————
# STREAMLIT UI
# ————————————————————————————————————————————————————————
st.title("📊 Reporte Mensual de Actividad")

uploaded = st.file_uploader("Selecciona tu CSV", type="csv")
if not uploaded:
    st.warning("Sube el archivo para continuar")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(uploaded.read())
    csv_path = tmp.name

st.header("Parámetros Generales")
c1, c2 = st.columns(2)
with c1:
    MESES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

    now_mty = datetime.now(ZoneInfo("America/Monterrey"))
    mes_index = now_mty.month - 1  # 0-11
    
    mes = st.selectbox("Mes", MESES, index=mes_index)
with c2:
    year = st.number_input("Año", 2000, 2100, 2025)

w1, w2 = st.columns(2)
with w1:
    work_start = st.time_input("Hora inicio", value=time(9,0), step=1800)
with w2:
    work_end   = st.time_input("Hora fin",   value=time(18,30), step=1800)

st.header("Parámetros Globales por Tabla")

def tabla_section(name):
    with st.expander(name):
        col1, col2, col3 = st.columns(3)
        with col1:
            picks_linea = st.number_input(f"{name} Picks en línea", 0, key=f"{name}_picks")
            num_guias   = st.number_input(f"{name} # Guías", 0, key=f"{name}_guias")
        with col2:
            movs_acab   = st.number_input(f"{name} Movs. acabados", 0, key=f"{name}_movs")
            mo_event    = st.number_input(f"{name} $MO Eventuales", 0.0, format="%.2f", key=f"{name}_mo_ev")
        with col3:
            costo_mo    = st.number_input(f"{name} Costo MO", 0.0, format="%.2f", key=f"{name}_costo")
        return {
            'picks_linea': picks_linea,
            'movimientos_acabados': movs_acab,
            'costo_mo': costo_mo,
            'num_guias': num_guias,
            'mo_eventuales': mo_event,
        }


params = {
    'FDA':  tabla_section('FDA'),
    'FABE': tabla_section('FABE'),
    'DQ':   tabla_section('DQ'),
}

# ——— Botones lado a lado: Ejecutar y Abrir Google Sheet
btn1, btn2 = st.columns(2)
with btn1:
    run_clicked = st.button("🚀 Ejecutar", use_container_width=True)
with btn2:
    sheet_url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit"
    if hasattr(st, "link_button"):
        st.link_button("🔗 Abrir Google Sheet", sheet_url, use_container_width=True)
    else:
        st.markdown(f"[🔗 Abrir Google Sheet]({sheet_url})")

if run_clicked:
    with st.spinner("Procesando…"):
        try:
            res = run_monthly_pipeline(
                path=csv_path,
                mes=mes,
                year=year,
                update_report=True,
                global_params=params,
                work_start=work_start,
                work_end=work_end
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()
    st.success("✅ ¡Reporte generado!")
    st.subheader("Métricas Mensuales")
    st.dataframe(res['metrics'])
    st.subheader("Filas Ignoradas")
    st.dataframe(res['ignored_rows'])
