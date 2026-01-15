from textwrap import dedent

from svd_parser import SvdIndex, _expand_dim_indices, _format_dim_name, _parse_int


def _write_svd(tmp_path):
    xml = dedent(
        """\
        <device>
          <name>TestDevice</name>
          <peripherals>
            <peripheral>
              <name>GPIO</name>
              <description>General Purpose IO</description>
              <baseAddress>0x40000000</baseAddress>
              <addressBlock>
                <offset>0x0</offset>
                <size>0x100</size>
              </addressBlock>
              <registers>
                <register>
                  <name>DATA</name>
                  <description>Data register</description>
                  <addressOffset>0x0</addressOffset>
                  <size>32</size>
                  <resetValue>0x1234</resetValue>
                  <fields>
                    <field>
                      <name>EN</name>
                      <bitOffset>0</bitOffset>
                      <bitWidth>1</bitWidth>
                      <enumeratedValues>
                        <enumeratedValue>
                          <name>DIS</name>
                          <value>0</value>
                        </enumeratedValue>
                        <enumeratedValue>
                          <name>EN</name>
                          <value>1</value>
                        </enumeratedValue>
                      </enumeratedValues>
                    </field>
                    <field>
                      <name>MODE</name>
                      <bitRange>[3:1]</bitRange>
                    </field>
                  </fields>
                </register>
                <register>
                  <name>BUF%s</name>
                  <addressOffset>0x10</addressOffset>
                  <dim>2</dim>
                  <dimIncrement>4</dimIncrement>
                  <dimIndex>0-1</dimIndex>
                  <size>16</size>
                </register>
                <cluster>
                  <name>CFG%s</name>
                  <addressOffset>0x20</addressOffset>
                  <dim>2</dim>
                  <dimIncrement>8</dimIncrement>
                  <dimIndex>A-B</dimIndex>
                  <register>
                    <name>CTRL</name>
                    <addressOffset>0x0</addressOffset>
                  </register>
                </cluster>
              </registers>
            </peripheral>
          </peripherals>
        </device>
        """
    )
    path = tmp_path / "test.svd"
    path.write_text(xml)
    return path


def test_dim_helpers():
    assert _expand_dim_indices("0-2", 3) == ["0", "1", "2"]
    assert _expand_dim_indices("A-B", 2) == ["A", "B"]
    assert _expand_dim_indices("1-Z", 1) == ["1-Z"]
    assert _expand_dim_indices("0", 3) == ["0", "1", "2"]
    assert _expand_dim_indices("", 3) == ["0", "1", "2"]
    assert _format_dim_name("REG%s", "1") == "REG1"
    assert _format_dim_name("REG%d", "2") == "REG2"
    assert _format_dim_name("REG", "3") == "REG3"
    assert _parse_int(5) == 5
    assert _parse_int("bad", default=7) == 7


def test_svd_parsing_and_lookup(tmp_path):
    path = _write_svd(tmp_path)
    svd = SvdIndex(str(path))

    assert svd.device_name == "TestDevice"
    assert len(svd.registers) == 5

    reg_data = svd.find_register(0x40000000)
    assert reg_data is not None
    assert reg_data.path == "GPIO.DATA"
    assert reg_data.reset_value == 0x1234

    reg_buf1 = svd.find_register(0x40000014)
    assert reg_buf1 is not None
    assert reg_buf1.name == "BUF1"
    assert reg_buf1.width_bytes == 2

    reg_cfga = svd.find_register(0x40000020)
    assert reg_cfga is not None
    assert reg_cfga.path == "GPIO.CFGA.CTRL"

    reg_cfgb = svd.find_register(0x40000028)
    assert reg_cfgb is not None
    assert reg_cfgb.path == "GPIO.CFGB.CTRL"

    block = svd.find_peripheral_block(0x40000040)
    assert block is not None
    assert block.name == "GPIO"

    en_field = reg_data.fields[0]
    assert en_field.name == "EN"
    assert en_field.mask == 0x1
    assert en_field.enums["EN"].value == 1

    mode_field = reg_data.fields[1]
    assert mode_field.lsb == 1
    assert mode_field.msb == 3
    assert svd.find_register(0x0) is None
    assert svd.find_peripheral_block(0x50000000) is None


def test_svd_bit_ranges_and_missing_nodes(tmp_path):
    xml = dedent(
        """\
        <device>
          <name>Demo</name>
          <peripherals>
            <peripheral>
              <name>PER</name>
              <baseAddress>0x50000000</baseAddress>
              <registers>
                <register>
                  <name>REG</name>
                  <addressOffset>0x0</addressOffset>
                  <fields>
                    <field>
                      <name>FIELD0</name>
                      <bitRange>[7-0]</bitRange>
                    </field>
                    <field>
                      <name>FIELD1</name>
                      <lsb>8</lsb>
                      <msb>9</msb>
                    </field>
                  </fields>
                </register>
              </registers>
            </peripheral>
          </peripherals>
        </device>
        """
    )
    path = tmp_path / "demo.svd"
    path.write_text(xml)
    svd = SvdIndex(str(path))
    reg = svd.find_register(0x50000000)
    assert reg is not None
    assert reg.fields[0].mask == 0xff
    assert svd.find_register(0x50000010) is None


def test_svd_parser_missing_peripherals(tmp_path):
    xml = "<device><name>Empty</name></device>"
    path = tmp_path / "empty.svd"
    path.write_text(xml)
    svd = SvdIndex(str(path))
    assert svd.find_register(0x0) is None
    assert svd.find_peripheral_block(0x0) is None


def test_svd_dim_and_enum_edge_cases(tmp_path):
    xml = dedent(
        """\
        <device>
          <name>Edge</name>
          <peripherals>
            <peripheral>
              <name>P</name>
              <baseAddress>0x40000000</baseAddress>
              <registers>
                <cluster>
                  <name>C%s</name>
                  <addressOffset>0x0</addressOffset>
                  <dim>2</dim>
                  <register>
                    <name>R</name>
                    <addressOffset>0x0</addressOffset>
                  </register>
                </cluster>
                <cluster>
                  <name>S</name>
                  <addressOffset>0x10</addressOffset>
                  <register>
                    <name>R2</name>
                    <addressOffset>0x0</addressOffset>
                  </register>
                </cluster>
                <register>
                  <name>REG%s</name>
                  <addressOffset>0x20</addressOffset>
                  <dim>2</dim>
                  <size>16</size>
                  <fields>
                    <field>
                      <name>ONEBIT</name>
                      <bitRange>[3]</bitRange>
                    </field>
                    <field>
                      <name>SKIP</name>
                    </field>
                    <field>
                      <name>ENUM</name>
                      <bitOffset>0</bitOffset>
                      <bitWidth>1</bitWidth>
                      <enumeratedValues>
                        <enumeratedValue>
                          <value>0</value>
                        </enumeratedValue>
                        <enumeratedValue>
                          <name>VAL</name>
                        </enumeratedValue>
                      </enumeratedValues>
                    </field>
                  </fields>
                </register>
              </registers>
            </peripheral>
          </peripherals>
        </device>
        """
    )
    path = tmp_path / "edge.svd"
    path.write_text(xml)
    svd = SvdIndex(str(path))
    assert svd.registers
    reg = svd.find_register(0x40000020)
    assert reg is not None
    assert any(field.name == "ONEBIT" for field in reg.fields)


def test_svd_peripheral_without_registers(tmp_path):
    xml = dedent(
        """\
        <device>
          <peripherals>
            <peripheral>
              <name>EMPTY</name>
              <baseAddress>0x0</baseAddress>
            </peripheral>
          </peripherals>
        </device>
        """
    )
    path = tmp_path / "noregs.svd"
    path.write_text(xml)
    svd = SvdIndex(str(path))
    assert svd.registers == []
