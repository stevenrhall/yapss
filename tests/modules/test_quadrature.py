"""

Test the yapss._private.quadrature module.

"""

# third party imports
import numpy as np
import pytest

# package imports
from yapss._private.quadrature import lgl, lgr

TOL = 0


def test_lgl():
    """Test that yapss.quadrature.lgl works properly.

    For n = 10, confirm that the results returned from lgl agree with precalculated values.
    """
    n = 10
    t, w, d, d0 = lgl(n)
    assert t == pytest.approx(t_lgl, rel=TOL, abs=0)
    assert w == pytest.approx(w_lgl, rel=TOL, abs=0)
    assert d == pytest.approx(d_lgl, rel=TOL, abs=0)
    assert d0 == pytest.approx(d0_lgl, rel=TOL, abs=0)


def test_lgr():
    """Test that yapss.quadrature.lgr works properly.

    For n = 10, confirm that the results returned from lgr agree with precalculated values.
    """
    n = 10
    t, w, d, b = lgr(n)
    assert t == pytest.approx(t_lgr, rel=TOL, abs=0)
    assert w == pytest.approx(w_lgr, rel=TOL, abs=0)
    assert d == pytest.approx(d_lgr, rel=TOL, abs=0)
    # TODO: test value of b


t_lgl = [
    -1.0000000000000000000,
    -0.9195339081664588138,
    -0.7387738651055050750,
    -0.4779249498104444957,
    -0.1652789576663870246,
    0.1652789576663870246,
    0.4779249498104444957,
    0.7387738651055050750,
    0.9195339081664588138,
    1.0000000000000000000,
]

w_lgl = [
    0.0222222222222222222,
    0.1333059908510701111,
    0.2248893420631264521,
    0.2920426836796837579,
    0.3275397611838974567,
    0.3275397611838974567,
    0.2920426836796837579,
    0.2248893420631264521,
    0.1333059908510701111,
    0.0222222222222222222,
]

d_lgl = np.array(
    [
        [
            -22.5000000000000000000,
            30.4381450292819186100,
            -12.1779467074298195480,
            6.9437884851339540050,
            -4.5993547611031327556,
            3.2946430337491841461,
            -2.4528841754426869369,
            1.8295639319032461325,
            -1.2759548360926636528,
            0.5000000000000000000,
        ],
        [
            -5.0740647029780635817,
            0.0000000000000000000,
            7.1855028697058254560,
            -3.3516638627467733272,
            2.0782079940364168961,
            -1.4449484487514552215,
            1.0591544636454412703,
            -0.7832392931379082838,
            0.5437537382357056024,
            -0.2127027580091888106,
        ],
        [
            1.2033519928522070063,
            -4.2592973549652187006,
            0.0000000000000000000,
            4.3686745570101854222,
            -2.1043501794131562387,
            1.3349154838782515123,
            -0.9366032131394473818,
            0.6767970871960860074,
            -0.4642749589081575577,
            0.1807865854892499307,
        ],
        [
            -0.5283693768202727060,
            1.5299026381816034045,
            -3.3641258682978176173,
            0.0000000000000000000,
            3.3873181012024447702,
            -1.6464940839870597055,
            1.0461893655024935459,
            -0.7212373127216038719,
            0.4834623263339481573,
            -0.1866457893937359772,
        ],
        [
            0.3120472556084112186,
            -0.8458135734064249801,
            1.4448503156016606950,
            -3.0202179581993472601,
            0.0000000000000000000,
            3.0251884877519746492,
            -1.4680555093899939238,
            0.9165551803364352536,
            -0.5880821430451694953,
            0.2235279447424538429,
        ],
        [
            -0.2235279447424538429,
            0.5880821430451694953,
            -0.9165551803364352536,
            1.4680555093899939238,
            -3.0251884877519746492,
            0.0000000000000000000,
            3.0202179581993472601,
            -1.4448503156016606950,
            0.8458135734064249801,
            -0.3120472556084112186,
        ],
        [
            0.1866457893937359772,
            -0.4834623263339481573,
            0.7212373127216038719,
            -1.0461893655024935459,
            1.6464940839870597055,
            -3.3873181012024447702,
            0.0000000000000000000,
            3.3641258682978176173,
            -1.5299026381816034045,
            0.5283693768202727060,
        ],
        [
            -0.1807865854892499307,
            0.4642749589081575577,
            -0.6767970871960860074,
            0.9366032131394473818,
            -1.3349154838782515123,
            2.1043501794131562387,
            -4.3686745570101854222,
            0.0000000000000000000,
            4.2592973549652187006,
            -1.2033519928522070063,
        ],
        [
            0.2127027580091888106,
            -0.5437537382357056024,
            0.7832392931379082838,
            -1.0591544636454412703,
            1.4449484487514552215,
            -2.0782079940364168961,
            3.3516638627467733272,
            -7.1855028697058254560,
            0.0000000000000000000,
            5.0740647029780635817,
        ],
        [
            -0.5000000000000000000,
            1.2759548360926636528,
            -1.8295639319032461325,
            2.4528841754426869369,
            -3.2946430337491841461,
            4.5993547611031327556,
            -6.9437884851339540050,
            12.1779467074298195480,
            -30.4381450292819186100,
            22.5000000000000000000,
        ],
    ],
)

d0_lgl = [
    -22.5000000000000000000,
    9.1865285180811617940,
    -7.0728072752333232048,
    6.2065905507465308486,
    -5.8606292353229183194,
    5.8606292353229183194,
    -6.2065905507465308486,
    7.0728072752333232048,
    -9.1865285180811617940,
    22.5000000000000000000,
]

t_lgr = [
    -1.0000000000000000000,
    -0.9274843742335810781,
    -0.7638420424200025996,
    -0.5256460303700792294,
    -0.2362344693905880493,
    0.0760591978379781302,
    0.3806648401447243659,
    0.6477666876740094363,
    0.8512252205816079107,
    0.9711751807022469027,
]

w_lgr = [
    0.0200000000000000000,
    0.1202966705574816315,
    0.2042701318790006756,
    0.2681948378411786961,
    0.3058592877244226210,
    0.3135824572269383767,
    0.2906101648329183111,
    0.2391934317143797134,
    0.1643760127369214757,
    0.0736170054867584989,
]

d_lgr = np.array(
    [
        [
            -25.2500000000000000000,
            34.4508365235765391270,
            -14.4101998999864853360,
            8.8388502530625614448,
            -6.5125393090947877776,
            5.4139924049122164465,
            -4.9614002845690224118,
            5.0010787448267737658,
            -5.6779989112748351242,
            8.1073804785470398654,
            -5.0000000000000000000,
        ],
        [
            -5.5199739752348020550,
            -0.2594054751799548386,
            8.3242585233966415409,
            -4.1765296555185323350,
            2.8803413340726295635,
            -2.3237328735219697963,
            2.0960608049284452529,
            -2.0940076290573763864,
            2.3654738139091717474,
            -3.3692039965169557205,
            2.0767191287227030271,
        ],
        [
            1.2442992124922412004,
            -4.4860397954613501151,
            -0.2834720955590767027,
            5.1723897190138511157,
            -2.7702999992981104019,
            2.0382290827563265946,
            -1.7587395673194122082,
            1.7154279672678490713,
            -1.9124555922236725790,
            2.7066331541163214302,
            -1.6659720857849674052,
        ],
        [
            -0.5028044875944284034,
            1.4827969362615042460,
            -3.4075307837881994540,
            -0.3277300173479387796,
            4.0991727487564514346,
            -2.3092540057844551340,
            1.8026733068558827815,
            -1.6749826194815164370,
            1.8208028135356269046,
            -2.5464639531463827059,
            1.5633200617334555471,
        ],
        [
            0.2632264541339698500,
            -0.7265840737598326697,
            1.2967347729227023238,
            -2.9125408665112349189,
            -0.4044540193467337251,
            3.7504226822585377137,
            -2.2323785262258766049,
            1.8741146946195574397,
            -1.9432603242626478667,
            2.6609745597467703282,
            -1.6262553535752118702,
        ],
        [
            -0.1595181225929870431,
            0.4273083023785219478,
            -0.6954894728673400068,
            1.1960813651865653732,
            -2.7339680077095622487,
            -0.5411602115958076399,
            3.8601197935967283829,
            -2.4741824575070613575,
            2.3275875667424979285,
            -3.0645919687331589663,
            1.8578132131016036301,
        ],
        [
            0.1057351053125139485,
            -0.2787927999362979987,
            0.4340718760652691302,
            -0.6753495175174541583,
            1.1770735353849139468,
            -2.7920508918113604654,
            -0.8073173176811703809,
            4.5039129382109728061,
            -3.2609690166219179177,
            3.9508062996947847796,
            -2.3571202111002536901,
        ],
        [
            -0.0736452718984733760,
            0.1924518890209365734,
            -0.2925491513103019191,
            0.4335988413825009283,
            -0.6828073182285264878,
            1.2365751448616024025,
            -3.1121192438744576953,
            -1.4195136646736352416,
            6.2693027565488669243,
            -5.9964600547833074873,
            3.4451660729547953786,
        ],
        [
            0.0513908946626396215,
            -0.1336200899422537337,
            0.2004598551154437144,
            -0.2897012841364307524,
            0.4351537007685599589,
            -0.7149974857202506723,
            1.3849133795287374224,
            -3.8532655491071273380,
            -3.3607846837660183959,
            12.6750915826595864830,
            -6.3946403200628863077,
        ],
        [
            -0.0317445372474820356,
            0.0823338633799671952,
            -0.1227334725699889654,
            0.1752760629496450345,
            -0.2577804588115768420,
            0.4072571495892480950,
            -0.7258706699087832669,
            1.5944180915047309205,
            -5.4833842237021485608,
            -17.3461625148496642960,
            21.7083907096660527210,
        ],
    ],
)
