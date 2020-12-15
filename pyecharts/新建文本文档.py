from pyecharts.faker import Faker
from pyecharts import options as opts 
from pyecharts.charts import Geo 
from pyecharts.globals import ChartType, SymbolType

c = (
        Geo()
        .add_schema(maptype="china")
        .add(
            "",
            [("深圳", 120), ("哈尔滨", 66), ("杭州", 77), ("重庆", 88), ("上海", 100), ("乌鲁木齐", 30),("北京", 30),("武汉",70)],
            type_=ChartType.EFFECT_SCATTER,
            color="green",
        )
        .add(
            "geo",
            [("北京", "上海"), ("武汉", "深圳"),("重庆", "杭州"),("哈尔滨", "重庆"),("乌鲁木齐", "哈尔滨"),("深圳", "乌鲁木齐"),("武汉", "北京")],
            type_=ChartType.LINES,
            effect_opts=opts.EffectOpts(
                symbol=SymbolType.ARROW, symbol_size=6, color="blue"
            ),
            linestyle_opts=opts.LineStyleOpts(curve=0.2),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="全国主要城市航班路线和数量"))
    )

c.render()# 显示地图
