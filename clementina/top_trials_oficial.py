import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

st = JournalStorage(JournalFileBackend("estudios/optuna_etapa_B_oficial.log"))
s = optuna.load_study(study_name="etapa_B_Memo_Fat_Short_Unet_FACL_oficial", storage=st)
done = [t for t in s.trials if t.state.name == "COMPLETE"]
for t in sorted(done, key=lambda t: t.value)[:5]:
    print(t.number, round(t.value, 6), t.params, t.user_attrs.get("n_parametros"))
